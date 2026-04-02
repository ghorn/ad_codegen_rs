use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Fields, GenericParam, Generics, Ident, Index, Type, parse_macro_input,
    parse_quote,
};

#[proc_macro_derive(Vectorize)]
pub fn derive_vectorize(input: TokenStream) -> TokenStream {
    match derive_vectorize_impl(parse_macro_input!(input as DeriveInput)) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn derive_vectorize_impl(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = input.ident;
    let generics = input.generics;
    let leaf_ident = extract_single_type_parameter(&generics)?;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let replacement: Type = parse_quote!(U);
    let output_ty = rebind_generics(&ident, &generics, &replacement);
    let output_expr = rebind_expr_path(&ident, &generics);
    let view_ident = format_ident!("{ident}View");
    let view_decl_generics = prepend_lifetime_to_generics(&generics, "'a");
    let view_use_generics = prepend_lifetime_to_generic_args(&generics, quote!('a));
    let view_self_generics = prepend_lifetime_to_generic_args(&generics, quote!('_));

    let Data::Struct(data) = input.data else {
        return Err(syn::Error::new_spanned(
            ident,
            "Vectorize can only be derived for structs",
        ));
    };

    let field_types = field_types(&data.fields);
    let mut where_predicates = where_clause.cloned().unwrap_or_else(|| parse_quote!(where));
    where_predicates
        .predicates
        .push(parse_quote!(#leaf_ident: ::optimization::ScalarLeaf));
    for field_ty in &field_types {
        if !is_leaf_type(field_ty, &leaf_ident) {
            where_predicates
                .predicates
                .push(parse_quote!(#field_ty: ::optimization::Vectorize<#leaf_ident>));
        }
    }
    let mut view_where_predicates = where_predicates.clone();
    view_where_predicates
        .predicates
        .push(parse_quote!(#leaf_ident: 'a));
    let flatten_statements = data.fields.iter().enumerate().map(|(index, field)| {
        let access = field_access(index, field.ident.as_ref());
        if is_leaf_type(&field.ty, &leaf_ident) {
            quote! { out.push(&self.#access); }
        } else {
            quote! {
                ::optimization::Vectorize::<#leaf_ident>::flatten_refs(&self.#access, out);
            }
        }
    });
    let construct_fields = data.fields.iter().enumerate().map(|(index, field)| {
        let access = field.ident.clone().map_or_else(
            || {
                let tuple_index = Index::from(index);
                quote!(#tuple_index)
            },
            |name| quote!(#name),
        );
        let value_expr = construct_value_expr(&field.ty, &leaf_ident);
        quote! { #access: #value_expr }
    });
    let view_field_defs = data.fields.iter().enumerate().map(|(index, field)| {
        let access = field.ident.clone().map_or_else(
            || {
                let tuple_index = Index::from(index);
                quote!(#tuple_index)
            },
            |name| quote!(#name),
        );
        let view_ty = if is_leaf_type(&field.ty, &leaf_ident) {
            quote!(&'a #leaf_ident)
        } else {
            let field_ty = &field.ty;
            quote!(<#field_ty as ::optimization::Vectorize<#leaf_ident>>::View<'a>)
        };
        quote! { pub #access: #view_ty }
    });
    let view_construct_fields = data.fields.iter().enumerate().map(|(index, field)| {
        let access = field.ident.clone().map_or_else(
            || {
                let tuple_index = Index::from(index);
                quote!(#tuple_index)
            },
            |name| quote!(#name),
        );
        let value_expr = construct_view_expr(&field.ty, &leaf_ident);
        quote! { #access: #value_expr }
    });
    let view_self_fields = data.fields.iter().enumerate().map(|(index, field)| {
        let access = field.ident.clone().map_or_else(
            || {
                let tuple_index = Index::from(index);
                quote!(#tuple_index)
            },
            |name| quote!(#name),
        );
        let value_expr = if is_leaf_type(&field.ty, &leaf_ident) {
            quote!(&self.#access)
        } else {
            let field_ty = &field.ty;
            quote!(<#field_ty as ::optimization::Vectorize<#leaf_ident>>::view(&self.#access))
        };
        quote! { #access: #value_expr }
    });
    let len_terms = field_types.iter().map(|field_ty| {
        if is_leaf_type(field_ty, &leaf_ident) {
            quote!(1usize)
        } else {
            quote!(<#field_ty as ::optimization::Vectorize<#leaf_ident>>::LEN)
        }
    });
    let construct_expr = match &data.fields {
        Fields::Named(_) => quote!(#output_expr { #(#construct_fields,)* }),
        Fields::Unnamed(_) => quote!(#output_expr ( #(#construct_fields,)* )),
        Fields::Unit => quote!(#output_expr),
    };
    let view_struct = match &data.fields {
        Fields::Named(_) => quote! {
            pub struct #view_ident #view_decl_generics
            #view_where_predicates
            {
                #(#view_field_defs,)*
            }
        },
        Fields::Unnamed(fields) => {
            let types = fields.unnamed.iter().map(|field| {
                if is_leaf_type(&field.ty, &leaf_ident) {
                    quote!(&'a #leaf_ident)
                } else {
                    let field_ty = &field.ty;
                    quote!(<#field_ty as ::optimization::Vectorize<#leaf_ident>>::View<'a>)
                }
            });
            quote! {
                pub struct #view_ident #view_decl_generics
                #view_where_predicates
                ( #(pub #types),* );
            }
        }
        Fields::Unit => quote! {
            pub struct #view_ident #view_decl_generics
            #view_where_predicates
            ;
        },
    };
    let view_construct_expr = match &data.fields {
        Fields::Named(_) => {
            quote!(#view_ident :: #view_use_generics { #(#view_construct_fields,)* })
        }
        Fields::Unnamed(_) => {
            quote!(#view_ident :: #view_use_generics ( #(#view_construct_fields,)* ))
        }
        Fields::Unit => quote!(#view_ident :: #view_use_generics),
    };
    let view_from_self_expr = match &data.fields {
        Fields::Named(_) => {
            quote!(#view_ident :: #view_self_generics { #(#view_self_fields,)* })
        }
        Fields::Unnamed(_) => {
            quote!(#view_ident :: #view_self_generics ( #(#view_self_fields,)* ))
        }
        Fields::Unit => quote!(#view_ident :: #view_self_generics),
    };
    Ok(quote! {
        #view_struct

        impl #impl_generics #ident #ty_generics
        #where_predicates
        {
            #[doc(hidden)]
            pub fn __optimization_from_flat<U: ::optimization::ScalarLeaf>(
                f: &mut impl ::core::ops::FnMut() -> U
            ) -> #output_ty {
                #construct_expr
            }
        }

        impl #impl_generics ::optimization::Vectorize<#leaf_ident> for #ident #ty_generics
        #where_predicates
        {
            type Rebind<U: ::optimization::ScalarLeaf> = #output_ty;
            type View<'a> = #view_ident #view_use_generics where #leaf_ident: 'a, Self: 'a;

            const LEN: usize = 0 #(+ #len_terms)*;

            fn flatten_refs<'a>(&'a self, out: &mut ::std::vec::Vec<&'a #leaf_ident>) {
                #(#flatten_statements)*
            }

            fn from_flat_fn<U: ::optimization::ScalarLeaf>(
                f: &mut impl ::core::ops::FnMut() -> U
            ) -> Self::Rebind<U> {
                Self::__optimization_from_flat::<U>(f)
            }

            fn view<'a>(&'a self) -> Self::View<'a>
            where
                Self: 'a,
                #leaf_ident: 'a,
            {
                #view_from_self_expr
            }

            fn view_from_flat_slice<'a>(
                slice: &'a [#leaf_ident],
                index: &mut usize
            ) -> Self::View<'a>
            where
                #leaf_ident: 'a,
            {
                #view_construct_expr
            }
        }
    })
}

fn extract_single_type_parameter(generics: &Generics) -> syn::Result<Ident> {
    let mut type_params = generics.type_params();
    let first = type_params.next().ok_or_else(|| {
        syn::Error::new_spanned(
            generics,
            "Vectorize requires exactly one generic type parameter",
        )
    })?;
    if type_params.next().is_some()
        || generics
            .params
            .iter()
            .any(|param| matches!(param, GenericParam::Lifetime(_)))
    {
        return Err(syn::Error::new_spanned(
            generics,
            "Vectorize requires exactly one generic type parameter and does not support lifetime parameters",
        ));
    }
    Ok(first.ident.clone())
}

fn field_types(fields: &Fields) -> Vec<Type> {
    fields
        .iter()
        .map(|field| field.ty.clone())
        .collect::<Vec<_>>()
}

fn field_access(index: usize, ident: Option<&Ident>) -> proc_macro2::TokenStream {
    ident.cloned().map_or_else(
        || {
            let tuple_index = Index::from(index);
            quote!(#tuple_index)
        },
        |name| quote!(#name),
    )
}

fn rebind_generics(
    ident: &Ident,
    generics: &Generics,
    replacement: &Type,
) -> proc_macro2::TokenStream {
    let args = generics.params.iter().map(|param| match param {
        GenericParam::Type(_) => quote!(#replacement),
        GenericParam::Lifetime(lifetime) => {
            let lifetime = &lifetime.lifetime;
            quote!(#lifetime)
        }
        GenericParam::Const(const_param) => {
            let ident = &const_param.ident;
            quote!(#ident)
        }
    });
    quote!(#ident < #(#args),* >)
}

fn rebind_expr_path(ident: &Ident, generics: &Generics) -> proc_macro2::TokenStream {
    let args = generics.params.iter().map(|param| match param {
        GenericParam::Type(_) => quote!(U),
        GenericParam::Lifetime(lifetime) => {
            let lifetime = &lifetime.lifetime;
            quote!(#lifetime)
        }
        GenericParam::Const(const_param) => {
            let ident = &const_param.ident;
            quote!(#ident)
        }
    });
    quote!(#ident :: < #(#args),* >)
}

fn prepend_lifetime_to_generics(generics: &Generics, lifetime: &str) -> proc_macro2::TokenStream {
    let lifetime: syn::Lifetime = syn::parse_str(lifetime).expect("valid lifetime");
    let params = generics.params.iter().map(|param| match param {
        GenericParam::Type(type_param) => {
            let ident = &type_param.ident;
            quote!(#ident)
        }
        GenericParam::Const(const_param) => {
            let ident = &const_param.ident;
            let ty = &const_param.ty;
            quote!(const #ident: #ty)
        }
        GenericParam::Lifetime(existing) => {
            let lt = &existing.lifetime;
            quote!(#lt)
        }
    });
    quote!(< #lifetime, #(#params),* >)
}

fn prepend_lifetime_to_generic_args(
    generics: &Generics,
    lifetime: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let args = generics.params.iter().map(|param| match param {
        GenericParam::Type(type_param) => {
            let ident = &type_param.ident;
            quote!(#ident)
        }
        GenericParam::Const(const_param) => {
            let ident = &const_param.ident;
            quote!(#ident)
        }
        GenericParam::Lifetime(existing) => {
            let lt = &existing.lifetime;
            quote!(#lt)
        }
    });
    quote!(< #lifetime, #(#args),* >)
}

fn is_leaf_type(ty: &Type, leaf_ident: &Ident) -> bool {
    match ty {
        Type::Path(path) => path.qself.is_none() && path.path.is_ident(leaf_ident),
        _ => false,
    }
}

fn construct_value_expr(ty: &Type, leaf_ident: &Ident) -> proc_macro2::TokenStream {
    if is_leaf_type(ty, leaf_ident) {
        return quote!(f());
    }
    if let Type::Array(array) = ty {
        let value_expr = construct_value_expr(&array.elem, leaf_ident);
        quote!(::std::array::from_fn(|_| #value_expr))
    } else {
        quote!(<#ty>::__optimization_from_flat::<U>(f))
    }
}

fn construct_view_expr(ty: &Type, leaf_ident: &Ident) -> proc_macro2::TokenStream {
    if is_leaf_type(ty, leaf_ident) {
        return quote!({
            let value = &slice[*index];
            *index += 1;
            value
        });
    }
    quote!(<#ty as ::optimization::Vectorize<#leaf_ident>>::view_from_flat_slice(slice, index))
}
