use proc_macro::TokenStream;
use quote::quote;
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

    let Data::Struct(data) = input.data else {
        return Err(syn::Error::new_spanned(
            ident,
            "Vectorize can only be derived for structs",
        ));
    };

    let field_types = field_types(&data.fields);
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

    Ok(quote! {
        impl #impl_generics ::optimization::Vectorize<#leaf_ident> for #ident #ty_generics
        #where_predicates
        {
            type Rebind<U: ::optimization::ScalarLeaf> = #output_ty;

            const LEN: usize = 0 #(+ #len_terms)*;

            fn flatten_refs<'a>(&'a self, out: &mut ::std::vec::Vec<&'a #leaf_ident>) {
                #(#flatten_statements)*
            }

            fn from_flat_fn<U: ::optimization::ScalarLeaf>(
                f: &mut impl ::core::ops::FnMut() -> U
            ) -> Self::Rebind<U> {
                #construct_expr
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
        quote!(<#ty as ::optimization::Vectorize<#leaf_ident>>::from_flat_fn::<U>(f))
    }
}
