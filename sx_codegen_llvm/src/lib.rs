use std::ffi::{CStr, CString};
use std::fs;
use std::mem;
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use llvm_sys::LLVMRealPredicate::{LLVMRealOGT, LLVMRealOLT};
use llvm_sys::analysis::{LLVMVerifierFailureAction, LLVMVerifyModule};
use llvm_sys::core::{
    LLVMAddFunction, LLVMAppendBasicBlockInContext, LLVMBuildCall2, LLVMBuildFAdd, LLVMBuildFCmp,
    LLVMBuildFDiv, LLVMBuildFMul, LLVMBuildFSub, LLVMBuildInBoundsGEP2, LLVMBuildLoad2,
    LLVMBuildRetVoid, LLVMBuildSelect, LLVMBuildStore, LLVMConstInt, LLVMConstReal,
    LLVMContextCreate, LLVMContextDispose, LLVMCreateBuilderInContext, LLVMDisposeBuilder,
    LLVMDisposeMemoryBuffer, LLVMDisposeMessage, LLVMDoubleTypeInContext, LLVMFunctionType,
    LLVMGetBufferSize, LLVMGetBufferStart, LLVMGetNamedFunction, LLVMGetParam,
    LLVMGlobalGetValueType, LLVMInt64TypeInContext, LLVMModuleCreateWithNameInContext,
    LLVMPointerType, LLVMPositionBuilderAtEnd, LLVMSetTarget, LLVMVoidTypeInContext,
};
use llvm_sys::error::{LLVMDisposeErrorMessage, LLVMErrorRef, LLVMGetErrorMessage};
use llvm_sys::orc2::LLVMOrcExecutorAddress;
use llvm_sys::orc2::lljit::{
    LLVMOrcCreateLLJIT, LLVMOrcDisposeLLJIT, LLVMOrcLLJITAddObjectFile,
    LLVMOrcLLJITGetGlobalPrefix, LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITLookup,
};
use llvm_sys::orc2::{
    LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess, LLVMOrcJITDylibAddGenerator,
};
use llvm_sys::prelude::{
    LLVMBuilderRef, LLVMContextRef, LLVMMemoryBufferRef, LLVMModuleRef, LLVMTypeRef, LLVMValueRef,
};
use llvm_sys::target::{
    LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget, LLVMCreateTargetData,
    LLVMDisposeTargetData, LLVMSetModuleDataLayout,
};
use llvm_sys::target_machine::{
    LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetMachineOptions,
    LLVMCreateTargetMachineWithOptions, LLVMDisposeTargetMachine, LLVMDisposeTargetMachineOptions,
    LLVMGetDefaultTargetTriple, LLVMGetHostCPUFeatures, LLVMGetHostCPUName,
    LLVMGetTargetFromTriple, LLVMTargetMachineEmitToMemoryBuffer, LLVMTargetMachineOptionsSetCPU,
    LLVMTargetMachineOptionsSetCodeGenOptLevel, LLVMTargetMachineOptionsSetCodeModel,
    LLVMTargetMachineOptionsSetFeatures, LLVMTargetMachineRef,
};
use llvm_sys::transforms::pass_builder::{
    LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
};
use sx_codegen::{
    InstructionKind, LoweredFunction, ValueRef, format_rust_source, lower_function, sanitize_ident,
    to_pascal_case,
};
use sx_core::{BinaryOp, SXFunction, UnaryOp};

type RawKernelFn = unsafe extern "C" fn(*const *const f64, *const *mut f64);

fn kernel_symbol_name(name: &str) -> String {
    format!("__sx_codegen_llvm_{}", sanitize_ident(name))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlvmOptimizationLevel {
    O0,
    O2,
    O3,
    Os,
}

impl LlvmOptimizationLevel {
    fn codegen_level(self) -> LLVMCodeGenOptLevel {
        match self {
            Self::O0 => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            Self::O2 | Self::Os => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
            Self::O3 => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        }
    }

    fn pass_pipeline(self) -> &'static str {
        match self {
            Self::O0 => "default<O0>",
            Self::O2 => "default<O2>",
            Self::O3 => "default<O3>",
            Self::Os => "default<Os>",
        }
    }

    pub fn from_cargo_opt_level(opt_level: &str) -> Option<Self> {
        match opt_level {
            "0" => Some(Self::O0),
            "2" => Some(Self::O2),
            "3" => Some(Self::O3),
            "s" | "z" => Some(Self::Os),
            _ => None,
        }
    }
}

pub type JitOptimizationLevel = LlvmOptimizationLevel;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum LlvmTarget {
    #[default]
    Native,
    Triple(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AotWrapperOptions {
    pub emit_doc_comments: bool,
}

#[derive(Debug)]
pub struct CompiledJitFunction {
    lowered: LoweredFunction,
    lljit: llvm_sys::orc2::lljit::LLVMOrcLLJITRef,
    function: RawKernelFn,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JitExecutionContext {
    inputs: Vec<Vec<f64>>,
    outputs: Vec<Vec<f64>>,
    input_ptrs: Vec<*const f64>,
    output_ptrs: Vec<*mut f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LlvmCompileMode {
    Jit,
    Aot,
}

impl CompiledJitFunction {
    pub fn compile_function(
        function: &SXFunction,
        opt_level: LlvmOptimizationLevel,
    ) -> Result<Self> {
        Self::compile_lowered(&lower_function(function)?, opt_level)
    }

    pub fn compile_lowered(
        lowered: &LoweredFunction,
        opt_level: LlvmOptimizationLevel,
    ) -> Result<Self> {
        ensure_native_llvm_initialized()?;

        let object = build_object_buffer(
            lowered,
            opt_level,
            &LlvmTarget::Native,
            LlvmCompileMode::Jit,
        )?;
        let lljit = create_lljit()?;
        let main_dylib = unsafe { LLVMOrcLLJITGetMainJITDylib(lljit) };
        let prefix = unsafe { LLVMOrcLLJITGetGlobalPrefix(lljit) };
        attach_current_process_symbols(main_dylib, prefix)?;
        let add_error = unsafe { LLVMOrcLLJITAddObjectFile(lljit, main_dylib, object) };
        if let Err(error) = consume_llvm_error(add_error) {
            let _ = unsafe { LLVMOrcDisposeLLJIT(lljit) };
            return Err(error.context("failed to add object file to LLJIT"));
        }

        let address = lookup_symbol_address(lljit, &kernel_symbol_name(&lowered.name))?;
        let addr = usize::try_from(address)
            .map_err(|_| anyhow!("JIT symbol address does not fit into usize"))?;
        let function = unsafe { mem::transmute::<usize, RawKernelFn>(addr) };

        Ok(Self {
            lowered: lowered.clone(),
            lljit,
            function,
        })
    }

    pub fn lowered(&self) -> &LoweredFunction {
        &self.lowered
    }

    pub fn create_context(&self) -> JitExecutionContext {
        JitExecutionContext::new(&self.lowered)
    }

    pub fn eval(&self, context: &mut JitExecutionContext) {
        unsafe {
            (self.function)(context.input_ptrs.as_ptr(), context.output_ptrs.as_ptr());
        }
    }
}

pub fn emit_object_file(
    path: impl AsRef<Path>,
    function: &SXFunction,
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
) -> Result<()> {
    emit_object_file_lowered(path, &lower_function(function)?, opt_level, target)
}

pub fn emit_object_file_lowered(
    path: impl AsRef<Path>,
    lowered: &LoweredFunction,
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
) -> Result<()> {
    fs::write(path, emit_object_bytes_lowered(lowered, opt_level, target)?).map_err(Into::into)
}

pub fn emit_object_bytes_lowered(
    lowered: &LoweredFunction,
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
) -> Result<Vec<u8>> {
    ensure_native_llvm_initialized()?;
    let object = build_object_buffer(lowered, opt_level, target, LlvmCompileMode::Aot)?;
    let bytes = unsafe { memory_buffer_to_bytes(object) }?;
    unsafe { LLVMDisposeMemoryBuffer(object) };
    Ok(bytes)
}

impl Drop for CompiledJitFunction {
    fn drop(&mut self) {
        let _ = unsafe { LLVMOrcDisposeLLJIT(self.lljit) };
    }
}

impl JitExecutionContext {
    fn new(lowered: &LoweredFunction) -> Self {
        let mut inputs = lowered
            .inputs
            .iter()
            .map(|slot| vec![0.0; slot.ccs.nnz()])
            .collect::<Vec<_>>();
        let mut outputs = lowered
            .outputs
            .iter()
            .map(|slot| vec![0.0; slot.ccs.nnz()])
            .collect::<Vec<_>>();
        let input_ptrs = inputs
            .iter_mut()
            .map(|slot| slot.as_ptr())
            .collect::<Vec<_>>();
        let output_ptrs = outputs
            .iter_mut()
            .map(|slot| slot.as_mut_ptr())
            .collect::<Vec<_>>();
        Self {
            inputs,
            outputs,
            input_ptrs,
            output_ptrs,
        }
    }

    pub fn input_mut(&mut self, slot: usize) -> &mut [f64] {
        &mut self.inputs[slot]
    }

    pub fn output(&self, slot: usize) -> &[f64] {
        &self.outputs[slot]
    }

    pub fn output_mut(&mut self, slot: usize) -> &mut [f64] {
        &mut self.outputs[slot]
    }
}

fn ensure_native_llvm_initialized() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| unsafe {
        if LLVM_InitializeNativeTarget() != 0 {
            return Err("LLVM failed to initialize the native target".into());
        }
        if LLVM_InitializeNativeAsmPrinter() != 0 {
            return Err("LLVM failed to initialize the native asm printer".into());
        }
        Ok(())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

fn build_object_buffer(
    lowered: &LoweredFunction,
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
    compile_mode: LlvmCompileMode,
) -> Result<LLVMMemoryBufferRef> {
    unsafe {
        let context = LLVMContextCreate();
        let (target_machine, triple) = create_target_machine(opt_level, target, compile_mode)?;
        let module = match build_module(lowered, context, target_machine, &triple) {
            Ok(module) => module,
            Err(error) => {
                LLVMDisposeTargetMachine(target_machine);
                LLVMContextDispose(context);
                return Err(error);
            }
        };

        let object = (|| -> Result<LLVMMemoryBufferRef> {
            verify_module(module, "before optimization")?;
            run_default_pass_pipeline(module, target_machine, opt_level)?;
            verify_module(module, "after optimization")?;
            emit_object_buffer(module, target_machine)
        })();

        llvm_sys::core::LLVMDisposeModule(module);
        LLVMDisposeTargetMachine(target_machine);
        LLVMContextDispose(context);
        object
    }
}

unsafe fn create_target_machine(
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
    compile_mode: LlvmCompileMode,
) -> Result<(LLVMTargetMachineRef, CString)> {
    let host_triple = unsafe { take_llvm_message(LLVMGetDefaultTargetTriple()) }?;
    let triple = match target {
        LlvmTarget::Native => host_triple.clone(),
        LlvmTarget::Triple(triple) => triple.clone(),
    };
    let use_host_cpu = match target {
        LlvmTarget::Native => true,
        LlvmTarget::Triple(target_triple) => target_triple == &host_triple,
    };
    let cpu = if use_host_cpu {
        unsafe { take_llvm_message(LLVMGetHostCPUName()) }?
    } else {
        "generic".to_string()
    };
    let features = if use_host_cpu {
        unsafe { take_llvm_message(LLVMGetHostCPUFeatures()) }?
    } else {
        String::new()
    };
    let triple_c = CString::new(triple)?;
    let cpu_c = CString::new(cpu)?;
    let features_c = CString::new(features)?;

    let mut target = ptr::null_mut();
    let mut error_message = ptr::null_mut();
    if unsafe { LLVMGetTargetFromTriple(triple_c.as_ptr(), &mut target, &mut error_message) } != 0 {
        let message = unsafe { take_owned_message(error_message) }?;
        bail!("failed to resolve LLVM target from triple: {message}");
    }

    let options = unsafe { LLVMCreateTargetMachineOptions() };
    if options.is_null() {
        bail!("LLVMCreateTargetMachineOptions returned null");
    }
    unsafe {
        LLVMTargetMachineOptionsSetCPU(options, cpu_c.as_ptr());
        LLVMTargetMachineOptionsSetFeatures(options, features_c.as_ptr());
        LLVMTargetMachineOptionsSetCodeGenOptLevel(options, opt_level.codegen_level());
        LLVMTargetMachineOptionsSetCodeModel(
            options,
            match compile_mode {
                LlvmCompileMode::Jit => LLVMCodeModel::LLVMCodeModelJITDefault,
                LlvmCompileMode::Aot => LLVMCodeModel::LLVMCodeModelDefault,
            },
        );
    }
    let target_machine =
        unsafe { LLVMCreateTargetMachineWithOptions(target, triple_c.as_ptr(), options) };
    unsafe { LLVMDisposeTargetMachineOptions(options) };
    if target_machine.is_null() {
        bail!("LLVMCreateTargetMachineWithOptions returned null");
    }
    Ok((target_machine, triple_c))
}

unsafe fn build_module(
    lowered: &LoweredFunction,
    context: LLVMContextRef,
    target_machine: LLVMTargetMachineRef,
    triple: &CString,
) -> Result<LLVMModuleRef> {
    let module_name = CString::new(format!("{}_llvm_jit", lowered.name))?;
    let symbol_name = CString::new(kernel_symbol_name(&lowered.name))?;
    let module = unsafe { LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context) };
    unsafe { LLVMSetTarget(module, triple.as_ptr()) };

    let layout_string = unsafe { llvm_data_layout_string(target_machine) }?;
    let layout_c = CString::new(layout_string)?;
    let layout = unsafe { LLVMCreateTargetData(layout_c.as_ptr()) };
    unsafe {
        LLVMSetModuleDataLayout(module, layout);
        LLVMDisposeTargetData(layout);
    }

    let builder = unsafe { LLVMCreateBuilderInContext(context) };
    let build_result = (|| -> Result<()> {
        let f64_ty = unsafe { LLVMDoubleTypeInContext(context) };
        let i64_ty = unsafe { LLVMInt64TypeInContext(context) };
        let f64_ptr_ty = unsafe { LLVMPointerType(f64_ty, 0) };
        let ptr_array_ty = unsafe { LLVMPointerType(f64_ptr_ty, 0) };
        let void_ty = unsafe { LLVMVoidTypeInContext(context) };
        let mut param_types = [ptr_array_ty, ptr_array_ty];
        let function_ty = unsafe {
            LLVMFunctionType(
                void_ty,
                param_types.as_mut_ptr(),
                param_types.len() as u32,
                0,
            )
        };
        let function = unsafe { LLVMAddFunction(module, symbol_name.as_ptr(), function_ty) };
        let entry = unsafe { LLVMAppendBasicBlockInContext(context, function, c"entry".as_ptr()) };
        unsafe { LLVMPositionBuilderAtEnd(builder, entry) };

        let inputs_param = unsafe { LLVMGetParam(function, 0) };
        let outputs_param = unsafe { LLVMGetParam(function, 1) };
        let mut temps = Vec::with_capacity(lowered.instructions.len());
        for instruction in &lowered.instructions {
            let temp = match instruction.kind {
                InstructionKind::Unary { op, input } => {
                    let input = unsafe {
                        emit_value(
                            builder,
                            inputs_param,
                            &temps,
                            input,
                            f64_ty,
                            f64_ptr_ty,
                            i64_ty,
                        )
                    };
                    unsafe { emit_unary_op(builder, module, op, input, f64_ty) }
                }
                InstructionKind::Binary { op, lhs, rhs } => {
                    let lhs = unsafe {
                        emit_value(
                            builder,
                            inputs_param,
                            &temps,
                            lhs,
                            f64_ty,
                            f64_ptr_ty,
                            i64_ty,
                        )
                    };
                    let rhs = unsafe {
                        emit_value(
                            builder,
                            inputs_param,
                            &temps,
                            rhs,
                            f64_ty,
                            f64_ptr_ty,
                            i64_ty,
                        )
                    };
                    unsafe { emit_binary_op(builder, module, op, lhs, rhs, f64_ty) }
                }
            };
            if temps.len() != instruction.temp {
                bail!("lowered temp order is not contiguous");
            };
            temps.push(temp);
        }

        for (slot_idx, values) in lowered.output_values.iter().enumerate() {
            let output_ptr =
                unsafe { load_slot_ptr(builder, outputs_param, slot_idx, f64_ptr_ty, i64_ty) };
            for (offset, value) in values.iter().copied().enumerate() {
                let value_ref = unsafe {
                    emit_value(
                        builder,
                        inputs_param,
                        &temps,
                        value,
                        f64_ty,
                        f64_ptr_ty,
                        i64_ty,
                    )
                };
                let cell_ptr = unsafe { gep_f64_ptr(builder, output_ptr, offset, f64_ty, i64_ty) };
                unsafe { LLVMBuildStore(builder, value_ref, cell_ptr) };
            }
        }

        unsafe { LLVMBuildRetVoid(builder) };
        Ok(())
    })();

    unsafe { LLVMDisposeBuilder(builder) };
    match build_result {
        Ok(()) => Ok(module),
        Err(error) => {
            unsafe { llvm_sys::core::LLVMDisposeModule(module) };
            Err(error)
        }
    }
}

unsafe fn emit_unary_op(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    op: UnaryOp,
    input: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    match op {
        UnaryOp::Abs => unsafe { call_unary_math(builder, module, c"fabs", input, f64_ty) },
        UnaryOp::Floor => unsafe { call_unary_math(builder, module, c"floor", input, f64_ty) },
        UnaryOp::Ceil => unsafe { call_unary_math(builder, module, c"ceil", input, f64_ty) },
        UnaryOp::Round => unsafe { call_unary_math(builder, module, c"round", input, f64_ty) },
        UnaryOp::Trunc => unsafe { call_unary_math(builder, module, c"trunc", input, f64_ty) },
        UnaryOp::Sqrt => unsafe { call_unary_math(builder, module, c"sqrt", input, f64_ty) },
        UnaryOp::Exp => unsafe { call_unary_math(builder, module, c"exp", input, f64_ty) },
        UnaryOp::Log => unsafe { call_unary_math(builder, module, c"log", input, f64_ty) },
        UnaryOp::Sin => unsafe { call_unary_math(builder, module, c"sin", input, f64_ty) },
        UnaryOp::Cos => unsafe { call_unary_math(builder, module, c"cos", input, f64_ty) },
        UnaryOp::Tan => unsafe { call_unary_math(builder, module, c"tan", input, f64_ty) },
        UnaryOp::Asin => unsafe { call_unary_math(builder, module, c"asin", input, f64_ty) },
        UnaryOp::Acos => unsafe { call_unary_math(builder, module, c"acos", input, f64_ty) },
        UnaryOp::Atan => unsafe { call_unary_math(builder, module, c"atan", input, f64_ty) },
        UnaryOp::Sinh => unsafe { call_unary_math(builder, module, c"sinh", input, f64_ty) },
        UnaryOp::Cosh => unsafe { call_unary_math(builder, module, c"cosh", input, f64_ty) },
        UnaryOp::Tanh => unsafe { call_unary_math(builder, module, c"tanh", input, f64_ty) },
        UnaryOp::Asinh => unsafe { call_unary_math(builder, module, c"asinh", input, f64_ty) },
        UnaryOp::Acosh => unsafe { call_unary_math(builder, module, c"acosh", input, f64_ty) },
        UnaryOp::Atanh => unsafe { call_unary_math(builder, module, c"atanh", input, f64_ty) },
        UnaryOp::Sign => unsafe { emit_sign(builder, input, f64_ty) },
    }
}

unsafe fn emit_binary_op(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    op: BinaryOp,
    lhs: LLVMValueRef,
    rhs: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    match op {
        BinaryOp::Add => unsafe { LLVMBuildFAdd(builder, lhs, rhs, c"".as_ptr()) },
        BinaryOp::Sub => unsafe { LLVMBuildFSub(builder, lhs, rhs, c"".as_ptr()) },
        BinaryOp::Mul => unsafe { LLVMBuildFMul(builder, lhs, rhs, c"".as_ptr()) },
        BinaryOp::Div => unsafe { LLVMBuildFDiv(builder, lhs, rhs, c"".as_ptr()) },
        BinaryOp::Pow => unsafe { call_binary_math(builder, module, c"pow", lhs, rhs, f64_ty) },
        BinaryOp::Atan2 => unsafe { call_binary_math(builder, module, c"atan2", lhs, rhs, f64_ty) },
        BinaryOp::Hypot => unsafe { call_binary_math(builder, module, c"hypot", lhs, rhs, f64_ty) },
        BinaryOp::Mod => unsafe { call_binary_math(builder, module, c"fmod", lhs, rhs, f64_ty) },
        BinaryOp::Copysign => unsafe {
            call_binary_math(builder, module, c"copysign", lhs, rhs, f64_ty)
        },
    }
}

unsafe fn emit_sign(
    builder: LLVMBuilderRef,
    input: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let zero = unsafe { LLVMConstReal(f64_ty, 0.0) };
    let one = unsafe { LLVMConstReal(f64_ty, 1.0) };
    let minus_one = unsafe { LLVMConstReal(f64_ty, -1.0) };
    let positive = unsafe { LLVMBuildFCmp(builder, LLVMRealOGT, input, zero, c"".as_ptr()) };
    let negative = unsafe { LLVMBuildFCmp(builder, LLVMRealOLT, input, zero, c"".as_ptr()) };
    let negative_or_zero =
        unsafe { LLVMBuildSelect(builder, negative, minus_one, zero, c"".as_ptr()) };
    unsafe { LLVMBuildSelect(builder, positive, one, negative_or_zero, c"".as_ptr()) }
}

unsafe fn call_unary_math(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    name: &CStr,
    input: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let (function, function_ty) = unsafe { get_or_declare_math_fn(module, name, 1, f64_ty) };
    let mut args = [input];
    unsafe {
        LLVMBuildCall2(
            builder,
            function_ty,
            function,
            args.as_mut_ptr(),
            1,
            c"".as_ptr(),
        )
    }
}

unsafe fn call_binary_math(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    name: &CStr,
    lhs: LLVMValueRef,
    rhs: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let (function, function_ty) = unsafe { get_or_declare_math_fn(module, name, 2, f64_ty) };
    let mut args = [lhs, rhs];
    unsafe {
        LLVMBuildCall2(
            builder,
            function_ty,
            function,
            args.as_mut_ptr(),
            2,
            c"".as_ptr(),
        )
    }
}

unsafe fn get_or_declare_math_fn(
    module: LLVMModuleRef,
    name: &CStr,
    arity: usize,
    f64_ty: LLVMTypeRef,
) -> (LLVMValueRef, LLVMTypeRef) {
    let existing = unsafe { LLVMGetNamedFunction(module, name.as_ptr()) };
    if !existing.is_null() {
        return (existing, unsafe { LLVMGlobalGetValueType(existing) });
    }
    let mut params = vec![f64_ty; arity];
    let function_ty = unsafe { LLVMFunctionType(f64_ty, params.as_mut_ptr(), arity as u32, 0) };
    (
        unsafe { LLVMAddFunction(module, name.as_ptr(), function_ty) },
        function_ty,
    )
}

unsafe fn llvm_data_layout_string(target_machine: LLVMTargetMachineRef) -> Result<String> {
    let data_layout =
        unsafe { llvm_sys::target_machine::LLVMCreateTargetDataLayout(target_machine) };
    let string_ptr = unsafe { llvm_sys::target::LLVMCopyStringRepOfTargetData(data_layout) };
    let string = unsafe { take_llvm_message(string_ptr) }?;
    unsafe { LLVMDisposeTargetData(data_layout) };
    Ok(string)
}

unsafe fn emit_value(
    builder: LLVMBuilderRef,
    inputs_param: LLVMValueRef,
    temps: &[LLVMValueRef],
    value: ValueRef,
    f64_ty: LLVMTypeRef,
    f64_ptr_ty: LLVMTypeRef,
    i64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    match value {
        ValueRef::Const(value) => unsafe { LLVMConstReal(f64_ty, value) },
        ValueRef::Temp(temp) => temps[temp],
        ValueRef::Input { slot, offset } => {
            let base_ptr =
                unsafe { load_slot_ptr(builder, inputs_param, slot, f64_ptr_ty, i64_ty) };
            let value_ptr = unsafe { gep_f64_ptr(builder, base_ptr, offset, f64_ty, i64_ty) };
            unsafe { LLVMBuildLoad2(builder, f64_ty, value_ptr, c"".as_ptr()) }
        }
    }
}

unsafe fn load_slot_ptr(
    builder: LLVMBuilderRef,
    slots_param: LLVMValueRef,
    slot: usize,
    f64_ptr_ty: LLVMTypeRef,
    i64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let mut index = [unsafe { LLVMConstInt(i64_ty, slot as u64, 0) }];
    let slot_ptr = unsafe {
        LLVMBuildInBoundsGEP2(
            builder,
            f64_ptr_ty,
            slots_param,
            index.as_mut_ptr(),
            1,
            c"".as_ptr(),
        )
    };
    unsafe { LLVMBuildLoad2(builder, f64_ptr_ty, slot_ptr, c"".as_ptr()) }
}

unsafe fn gep_f64_ptr(
    builder: LLVMBuilderRef,
    base_ptr: LLVMValueRef,
    offset: usize,
    f64_ty: LLVMTypeRef,
    i64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let mut index = [unsafe { LLVMConstInt(i64_ty, offset as u64, 0) }];
    unsafe {
        LLVMBuildInBoundsGEP2(
            builder,
            f64_ty,
            base_ptr,
            index.as_mut_ptr(),
            1,
            c"".as_ptr(),
        )
    }
}

unsafe fn run_default_pass_pipeline(
    module: LLVMModuleRef,
    target_machine: LLVMTargetMachineRef,
    opt_level: JitOptimizationLevel,
) -> Result<()> {
    let options = unsafe { LLVMCreatePassBuilderOptions() };
    let passes = CString::new(opt_level.pass_pipeline())?;
    let result = consume_llvm_error(unsafe {
        LLVMRunPasses(module, passes.as_ptr(), target_machine, options)
    });
    unsafe { LLVMDisposePassBuilderOptions(options) };
    result.context("failed to optimize LLVM module")
}

unsafe fn emit_object_buffer(
    module: LLVMModuleRef,
    target_machine: LLVMTargetMachineRef,
) -> Result<LLVMMemoryBufferRef> {
    let mut error_message = ptr::null_mut();
    let mut buffer = ptr::null_mut();
    if unsafe {
        LLVMTargetMachineEmitToMemoryBuffer(
            target_machine,
            module,
            LLVMCodeGenFileType::LLVMObjectFile,
            &mut error_message,
            &mut buffer,
        )
    } != 0
    {
        let message = unsafe { take_owned_message(error_message) }?;
        bail!("failed to emit LLVM object file: {message}");
    }
    Ok(buffer)
}

unsafe fn memory_buffer_to_bytes(buffer: LLVMMemoryBufferRef) -> Result<Vec<u8>> {
    let start = unsafe { LLVMGetBufferStart(buffer) };
    if start.is_null() {
        bail!("LLVM returned a null object buffer");
    }
    let size = unsafe { LLVMGetBufferSize(buffer) };
    let bytes = unsafe { slice::from_raw_parts(start.cast::<u8>(), size) };
    Ok(bytes.to_vec())
}

pub fn generate_aot_wrapper_module(
    lowered: &LoweredFunction,
    options: &AotWrapperOptions,
) -> Result<String> {
    let mut out = String::new();
    let module_name = format!("{}_llvm_aot", sanitize_ident(&lowered.name));
    let context_name = format!("{}LlvmAotContext", to_pascal_case(&lowered.name));
    let has_context = lowered
        .inputs
        .iter()
        .chain(lowered.outputs.iter())
        .any(|slot| !slot.ccs.is_scalar());

    out.push_str(&format!("pub mod {module_name} {{\n"));
    if options.emit_doc_comments {
        out.push_str(&format!(
            "    //! Generated LLVM AOT wrapper for `{}`.\n",
            lowered.name
        ));
    }
    out.push_str(
        "    #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n    pub struct GeneratedCCS { pub nrow: usize, pub ncol: usize, pub col_ptrs: &'static [usize], pub row_indices: &'static [usize] }\n",
    );
    out.push_str(
        "    #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n    pub struct SlotSpec { pub name: &'static str, pub ccs: GeneratedCCS, pub nnz: usize }\n",
    );

    for slot in lowered.inputs.iter().chain(lowered.outputs.iter()) {
        emit_ccs(slot.name.as_str(), &slot.ccs, &mut out);
    }

    out.push_str("    pub const INPUT_SPECS: &[SlotSpec] = &[\n");
    for slot in &lowered.inputs {
        let upper = sanitize_ident(slot.name.as_str()).to_ascii_uppercase();
        out.push_str(&format!(
            "        SlotSpec {{ name: \"{}\", ccs: {upper}_CCS, nnz: {} }},\n",
            slot.name,
            slot.ccs.nnz()
        ));
    }
    out.push_str("    ];\n");
    out.push_str("    pub const OUTPUT_SPECS: &[SlotSpec] = &[\n");
    for slot in &lowered.outputs {
        let upper = sanitize_ident(slot.name.as_str()).to_ascii_uppercase();
        out.push_str(&format!(
            "        SlotSpec {{ name: \"{}\", ccs: {upper}_CCS, nnz: {} }},\n",
            slot.name,
            slot.ccs.nnz()
        ));
    }
    out.push_str("    ];\n");
    out.push_str("    unsafe extern \"C\" {\n");
    out.push_str(&format!(
        "        fn {}(inputs: *const *const f64, outputs: *const *mut f64);\n",
        kernel_symbol_name(&lowered.name)
    ));
    out.push_str("    }\n");

    let mut params = Vec::new();
    for input in &lowered.inputs {
        let ident = sanitize_ident(input.name.as_str());
        let ty = if input.ccs.is_scalar() {
            "f64"
        } else {
            "&[f64]"
        };
        params.push(format!("{ident}: {ty}"));
    }
    for output in &lowered.outputs {
        let ident = sanitize_ident(output.name.as_str());
        let ty = if output.ccs.is_scalar() {
            "&mut f64"
        } else {
            "&mut [f64]"
        };
        params.push(format!("{ident}: {ty}"));
    }
    out.push_str(&format!("    pub fn eval({}) {{\n", params.join(", ")));
    for input in &lowered.inputs {
        if !input.ccs.is_scalar() {
            let ident = sanitize_ident(input.name.as_str());
            out.push_str(&format!(
                "        debug_assert_eq!({ident}.len(), {});\n",
                input.ccs.nnz()
            ));
        }
    }
    for output in &lowered.outputs {
        if !output.ccs.is_scalar() {
            let ident = sanitize_ident(output.name.as_str());
            out.push_str(&format!(
                "        debug_assert_eq!({ident}.len(), {});\n",
                output.ccs.nnz()
            ));
        }
    }
    let mut input_ptrs = Vec::new();
    for input in &lowered.inputs {
        let ident = sanitize_ident(input.name.as_str());
        if input.ccs.is_scalar() {
            let buffer_ident = format!("{ident}_buffer");
            out.push_str(&format!("        let {buffer_ident} = [{ident}];\n"));
            input_ptrs.push(format!("{buffer_ident}.as_ptr()"));
        } else {
            input_ptrs.push(format!("{ident}.as_ptr()"));
        }
    }
    let mut output_ptrs = Vec::new();
    let mut scalar_output_assignments = Vec::new();
    for output in &lowered.outputs {
        let ident = sanitize_ident(output.name.as_str());
        if output.ccs.is_scalar() {
            let buffer_ident = format!("{ident}_buffer");
            out.push_str(&format!("        let mut {buffer_ident} = [0.0_f64];\n"));
            output_ptrs.push(format!("{buffer_ident}.as_mut_ptr()"));
            scalar_output_assignments.push(format!("        *{ident} = {buffer_ident}[0];\n"));
        } else {
            output_ptrs.push(format!("{ident}.as_mut_ptr()"));
        }
    }
    out.push_str(&format!(
        "        let input_ptrs: [*const f64; {}] = [{}];\n",
        input_ptrs.len(),
        input_ptrs.join(", ")
    ));
    out.push_str(&format!(
        "        let mut output_ptrs: [*mut f64; {}] = [{}];\n",
        output_ptrs.len(),
        output_ptrs.join(", ")
    ));
    out.push_str(&format!(
        "        unsafe {{ {}(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()); }}\n",
        kernel_symbol_name(&lowered.name)
    ));
    for assignment in scalar_output_assignments {
        out.push_str(&assignment);
    }
    out.push_str("    }\n");

    if has_context {
        out.push_str("    #[derive(Clone, Debug, PartialEq)]\n");
        out.push_str(&format!("    pub struct {context_name} {{\n"));
        for input in &lowered.inputs {
            let ident = sanitize_ident(input.name.as_str());
            let field_ty = if input.ccs.is_scalar() {
                "f64"
            } else {
                "Vec<f64>"
            };
            out.push_str(&format!("        pub {ident}: {field_ty},\n"));
        }
        for output in &lowered.outputs {
            let ident = sanitize_ident(output.name.as_str());
            let field_ty = if output.ccs.is_scalar() {
                "f64"
            } else {
                "Vec<f64>"
            };
            out.push_str(&format!("        pub {ident}: {field_ty},\n"));
        }
        out.push_str("    }\n");
        out.push_str(&format!("    impl Default for {context_name} {{\n"));
        out.push_str("        fn default() -> Self {\n");
        out.push_str("            Self {\n");
        for input in &lowered.inputs {
            let ident = sanitize_ident(input.name.as_str());
            if input.ccs.is_scalar() {
                out.push_str(&format!("                {ident}: 0.0,\n"));
            } else {
                out.push_str(&format!(
                    "                {ident}: vec![0.0; {}],\n",
                    input.ccs.nnz()
                ));
            }
        }
        for output in &lowered.outputs {
            let ident = sanitize_ident(output.name.as_str());
            if output.ccs.is_scalar() {
                out.push_str(&format!("                {ident}: 0.0,\n"));
            } else {
                out.push_str(&format!(
                    "                {ident}: vec![0.0; {}],\n",
                    output.ccs.nnz()
                ));
            }
        }
        out.push_str("            }\n");
        out.push_str("        }\n");
        out.push_str("    }\n");
        out.push_str(&format!("    impl {context_name} {{\n"));
        out.push_str("        pub fn new() -> Self { Self::default() }\n");
        out.push_str("        pub fn eval(&mut self) {\n");
        let mut call_args = Vec::new();
        for input in &lowered.inputs {
            let ident = sanitize_ident(input.name.as_str());
            if input.ccs.is_scalar() {
                call_args.push(format!("self.{ident}"));
            } else {
                call_args.push(format!("&self.{ident}"));
            }
        }
        for output in &lowered.outputs {
            let ident = sanitize_ident(output.name.as_str());
            call_args.push(format!("&mut self.{ident}"));
        }
        out.push_str(&format!("            eval({});\n", call_args.join(", ")));
        out.push_str("        }\n");
        out.push_str("    }\n");
    }

    out.push_str("}\n");
    format_rust_source(&out)
}

fn emit_ccs(name: &str, ccs: &sx_core::CCS, out: &mut String) {
    let upper = sanitize_ident(name).to_ascii_uppercase();
    let col_ptrs = format!("{upper}_COL_PTRS");
    let row_indices = format!("{upper}_ROW_INDICES");
    out.push_str(&format!(
        "    pub const {col_ptrs}: &[usize] = &{:?};\n",
        ccs.col_ptrs()
    ));
    out.push_str(&format!(
        "    pub const {row_indices}: &[usize] = &{:?};\n",
        ccs.row_indices()
    ));
    out.push_str(&format!(
        "    pub const {upper}_CCS: GeneratedCCS = GeneratedCCS {{ nrow: {}, ncol: {}, col_ptrs: {col_ptrs}, row_indices: {row_indices} }};\n",
        ccs.nrow(),
        ccs.ncol()
    ));
}

fn create_lljit() -> Result<llvm_sys::orc2::lljit::LLVMOrcLLJITRef> {
    let mut lljit = ptr::null_mut();
    let error = unsafe { LLVMOrcCreateLLJIT(&mut lljit, ptr::null_mut()) };
    consume_llvm_error(error).context("failed to create LLJIT")?;
    Ok(lljit)
}

fn attach_current_process_symbols(
    jit_dylib: llvm_sys::orc2::LLVMOrcJITDylibRef,
    global_prefix: i8,
) -> Result<()> {
    let mut generator = ptr::null_mut();
    let error = unsafe {
        LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
            &mut generator,
            global_prefix,
            None,
            ptr::null_mut(),
        )
    };
    consume_llvm_error(error).context("failed to create current-process search generator")?;
    unsafe { LLVMOrcJITDylibAddGenerator(jit_dylib, generator) };
    Ok(())
}

fn lookup_symbol_address(
    lljit: llvm_sys::orc2::lljit::LLVMOrcLLJITRef,
    symbol_name: &str,
) -> Result<LLVMOrcExecutorAddress> {
    let symbol = CString::new(symbol_name)?;
    let mut address = 0_u64;
    let first_attempt = unsafe { LLVMOrcLLJITLookup(lljit, &mut address, symbol.as_ptr()) };
    if first_attempt.is_null() {
        return Ok(address);
    }
    let first_error = llvm_error_message(first_attempt);
    let prefix = unsafe { LLVMOrcLLJITGetGlobalPrefix(lljit) };
    if prefix == 0 {
        return Err(first_error.context("failed to resolve JIT symbol"));
    }
    let prefixed_name = format!("{}{}", prefix as u8 as char, symbol_name);
    let prefixed = CString::new(prefixed_name)?;
    let retry = unsafe { LLVMOrcLLJITLookup(lljit, &mut address, prefixed.as_ptr()) };
    if retry.is_null() {
        Ok(address)
    } else {
        Err(llvm_error_message(retry).context(format!(
            "failed to resolve JIT symbol after retry; first lookup error: {}",
            first_error
        )))
    }
}

unsafe fn verify_module(module: LLVMModuleRef, phase: &str) -> Result<()> {
    let mut message = ptr::null_mut();
    if unsafe {
        LLVMVerifyModule(
            module,
            LLVMVerifierFailureAction::LLVMReturnStatusAction,
            &mut message,
        )
    } == 0
    {
        return Ok(());
    }
    let detail = unsafe { take_owned_message(message) }?;
    bail!("LLVM module verification failed {phase}: {detail}");
}

fn consume_llvm_error(error: LLVMErrorRef) -> Result<()> {
    if error.is_null() {
        Ok(())
    } else {
        Err(llvm_error_message(error))
    }
}

fn llvm_error_message(error: LLVMErrorRef) -> anyhow::Error {
    unsafe {
        let message = LLVMGetErrorMessage(error);
        let text = if message.is_null() {
            "unknown LLVM error".to_string()
        } else {
            let text = CStr::from_ptr(message).to_string_lossy().into_owned();
            LLVMDisposeErrorMessage(message);
            text
        };
        anyhow!(text)
    }
}

unsafe fn take_llvm_message(message: *mut i8) -> Result<String> {
    unsafe { take_owned_message(message) }
}

unsafe fn take_owned_message(message: *mut i8) -> Result<String> {
    if message.is_null() {
        bail!("LLVM returned a null message pointer");
    }
    let text = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned();
    unsafe { LLVMDisposeMessage(message) };
    Ok(text)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{
        AotWrapperOptions, CompiledJitFunction, JitOptimizationLevel, LlvmTarget,
        emit_object_bytes_lowered, generate_aot_wrapper_module,
    };
    use sx_codegen::lower_function;
    use sx_core::{BinaryOp, NamedMatrix, SXFunction, SXMatrix, UnaryOp};

    fn eval_lowered(lowered: &sx_codegen::LoweredFunction, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut temps = vec![0.0; lowered.instructions.len()];
        let resolve = |value: sx_codegen::ValueRef, temps: &[f64]| match value {
            sx_codegen::ValueRef::Const(value) => value,
            sx_codegen::ValueRef::Temp(temp) => temps[temp],
            sx_codegen::ValueRef::Input { slot, offset } => inputs[slot][offset],
        };
        for instruction in &lowered.instructions {
            temps[instruction.temp] = match instruction.kind {
                sx_codegen::InstructionKind::Unary { op, input } => {
                    let input = resolve(input, &temps);
                    match op {
                        UnaryOp::Abs => input.abs(),
                        UnaryOp::Sign => {
                            if input > 0.0 {
                                1.0
                            } else if input < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        }
                        UnaryOp::Floor => input.floor(),
                        UnaryOp::Ceil => input.ceil(),
                        UnaryOp::Round => input.round(),
                        UnaryOp::Trunc => input.trunc(),
                        UnaryOp::Sqrt => input.sqrt(),
                        UnaryOp::Exp => input.exp(),
                        UnaryOp::Log => input.ln(),
                        UnaryOp::Sin => input.sin(),
                        UnaryOp::Cos => input.cos(),
                        UnaryOp::Tan => input.tan(),
                        UnaryOp::Asin => input.asin(),
                        UnaryOp::Acos => input.acos(),
                        UnaryOp::Atan => input.atan(),
                        UnaryOp::Sinh => input.sinh(),
                        UnaryOp::Cosh => input.cosh(),
                        UnaryOp::Tanh => input.tanh(),
                        UnaryOp::Asinh => input.asinh(),
                        UnaryOp::Acosh => input.acosh(),
                        UnaryOp::Atanh => input.atanh(),
                    }
                }
                sx_codegen::InstructionKind::Binary { op, lhs, rhs } => {
                    let lhs = resolve(lhs, &temps);
                    let rhs = resolve(rhs, &temps);
                    match op {
                        BinaryOp::Add => lhs + rhs,
                        BinaryOp::Sub => lhs - rhs,
                        BinaryOp::Mul => lhs * rhs,
                        BinaryOp::Div => lhs / rhs,
                        BinaryOp::Pow => lhs.powf(rhs),
                        BinaryOp::Atan2 => lhs.atan2(rhs),
                        BinaryOp::Hypot => lhs.hypot(rhs),
                        BinaryOp::Mod => lhs % rhs,
                        BinaryOp::Copysign => lhs.copysign(rhs),
                    }
                }
            };
        }
        lowered
            .output_values
            .iter()
            .map(|slot| {
                slot.iter()
                    .copied()
                    .map(|value| resolve(value, &temps))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    #[test]
    fn llvm_jit_matches_lowered_reference_for_multi_io_kernel() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let y = SXMatrix::sym_dense("y", 1, 1).unwrap();
        let output0 = SXMatrix::scalar((x.nz(0) + y.nz(0)) / (1.0 + x.nz(1)));
        let output1 = SXMatrix::dense_column(vec![x.nz(0) * x.nz(1), x.nz(0) - y.nz(0)]).unwrap();
        let function = SXFunction::new(
            "llvm_demo",
            vec![
                NamedMatrix::new("x", x).unwrap(),
                NamedMatrix::new("y", y).unwrap(),
            ],
            vec![
                NamedMatrix::new("objective", output0).unwrap(),
                NamedMatrix::new("vector_out", output1).unwrap(),
            ],
        )
        .unwrap();

        let lowered = lower_function(&function).unwrap();
        let compiled =
            CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O3).unwrap();
        let mut context = compiled.create_context();
        context.input_mut(0).copy_from_slice(&[1.5, 0.25]);
        context.input_mut(1).copy_from_slice(&[0.75]);
        compiled.eval(&mut context);

        let expected = eval_lowered(&lowered, &[vec![1.5, 0.25], vec![0.75]]);
        assert_eq!(context.output(0).len(), expected[0].len());
        assert_eq!(context.output(1).len(), expected[1].len());
        for (lhs, rhs) in context.output(0).iter().zip(expected[0].iter()) {
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        for (lhs, rhs) in context.output(1).iter().zip(expected[1].iter()) {
            assert!((lhs - rhs).abs() <= 1e-12);
        }
    }

    #[test]
    fn llvm_aot_object_emission_produces_nonempty_bytes() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let value = SXMatrix::scalar(x.nz(0) * x.nz(1) + x.nz(0));
        let function = SXFunction::new(
            "llvm_aot_object",
            vec![NamedMatrix::new("x", x).unwrap()],
            vec![NamedMatrix::new("value", value).unwrap()],
        )
        .unwrap();
        let lowered = lower_function(&function).unwrap();
        let bytes =
            emit_object_bytes_lowered(&lowered, JitOptimizationLevel::O2, &LlvmTarget::Native)
                .unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn llvm_aot_wrapper_generation_includes_module_and_context() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let output = SXMatrix::dense_column(vec![x.nz(0) + 1.0, x.nz(1) / 2.0]).unwrap();
        let function = SXFunction::new(
            "llvm_aot_wrapper",
            vec![NamedMatrix::new("x", x).unwrap()],
            vec![NamedMatrix::new("output", output).unwrap()],
        )
        .unwrap();
        let lowered = lower_function(&function).unwrap();
        let wrapper = generate_aot_wrapper_module(
            &lowered,
            &AotWrapperOptions {
                emit_doc_comments: true,
            },
        )
        .unwrap();

        assert!(wrapper.contains("pub mod llvm_aot_wrapper_llvm_aot"));
        assert!(wrapper.contains("unsafe extern \"C\""));
        assert!(wrapper.contains("fn __sx_codegen_llvm_llvm_aot_wrapper"));
        assert!(wrapper.contains("pub struct LlvmAotWrapperLlvmAotContext"));
    }

    #[test]
    fn llvm_jit_matches_lowered_reference_for_transcendentals() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let output = SXMatrix::scalar(
            x.nz(0).sin()
                + x.nz(0).cos()
                + x.nz(0).tan()
                + x.nz(0).exp()
                + x.nz(0).log()
                + x.nz(0).sqrt()
                + x.nz(0).atan()
                + x.nz(0).sinh()
                + x.nz(0).tanh()
                + x.nz(0).powf(1.5)
                + x.nz(0).hypot(x.nz(1))
                + x.nz(0).atan2(x.nz(1))
                + x.nz(0).modulo(x.nz(1)),
        );
        let function = SXFunction::new(
            "llvm_transcendentals",
            vec![NamedMatrix::new("x", x).unwrap()],
            vec![NamedMatrix::new("y", output).unwrap()],
        )
        .unwrap();
        let lowered = lower_function(&function).unwrap();
        let compiled =
            CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O2).unwrap();
        for point in [[1.7, 0.8], [0.9, 1.1], [2.3, 0.6]] {
            let expected = eval_lowered(&lowered, &[point.to_vec()]);
            let mut context = compiled.create_context();
            context.input_mut(0).copy_from_slice(&point);
            compiled.eval(&mut context);
            assert_eq!(context.output(0).len(), 1);
            assert!((context.output(0)[0] - expected[0][0]).abs() <= 1e-12);
        }
    }

    #[test]
    fn llvm_jit_matches_lowered_reference_for_transcendental_gradient_bundle() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let objective = SXMatrix::scalar(
            x.nz(0).sin()
                + x.nz(0).exp()
                + x.nz(0).powf(1.5)
                + x.nz(0).hypot(x.nz(1))
                + x.nz(0).atan2(x.nz(1))
                + x.nz(1).log(),
        );
        let gradient = objective.gradient(&x).unwrap();
        let function = SXFunction::new(
            "llvm_transcendental_gradient_bundle",
            vec![NamedMatrix::new("x", x).unwrap()],
            vec![
                NamedMatrix::new("objective", objective).unwrap(),
                NamedMatrix::new("gradient", gradient).unwrap(),
            ],
        )
        .unwrap();
        let lowered = lower_function(&function).unwrap();
        let compiled =
            CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O2).unwrap();

        for point in [[1.7, 0.8], [0.9, 1.1], [2.3, 0.6]] {
            let expected = eval_lowered(&lowered, &[point.to_vec()]);
            let mut context = compiled.create_context();
            context.input_mut(0).copy_from_slice(&point);
            compiled.eval(&mut context);
            assert_eq!(context.output(0).len(), expected[0].len());
            assert_eq!(context.output(1).len(), expected[1].len());
            for (lhs, rhs) in context.output(0).iter().zip(expected[0].iter()) {
                assert!((lhs - rhs).abs() <= 1e-12);
            }
            for (lhs, rhs) in context.output(1).iter().zip(expected[1].iter()) {
                assert!((lhs - rhs).abs() <= 1e-12);
            }
        }
    }
}
