mod target;

struct MultiversionVersion {
    target: String,
    import: bool,
}

impl syn::parse::Parse for MultiversionVersion {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let lookahead1 = input.lookahead1();
        if lookahead1.peek(syn::Token![@]) {
            let _: syn::Token![@] = input.parse()?;
            let target: syn::LitStr = input.parse()?;
            Ok(Self {
                target: target.value(),
                import: true,
            })
        } else {
            let target: syn::LitStr = input.parse()?;
            Ok(Self {
                target: target.value(),
                import: false,
            })
        }
    }
}

struct Multiversion {
    versions: syn::punctuated::Punctuated<MultiversionVersion, syn::Token![,]>,
}

impl syn::parse::Parse for Multiversion {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Multiversion {
            versions: syn::punctuated::Punctuated::parse_terminated(input)?,
        })
    }
}

#[proc_macro_attribute]
pub fn multiversion(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let attr = syn::parse_macro_input!(attr as Multiversion);
    let item_fn = syn::parse::<syn::ItemFn>(item).expect("not a function item");
    let syn::ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = item_fn;
    let name = sig.ident.to_string();
    if sig.constness.is_some() {
        panic!("const functions are not supported");
    }
    if sig.asyncness.is_some() {
        panic!("async functions are not supported");
    }
    let generics_params = sig.generics.params.clone();
    for generic_param in generics_params.iter() {
        if !matches!(generic_param, syn::GenericParam::Lifetime(_)) {
            panic!("generic parameters are not supported");
        }
    }
    let generics_where = sig.generics.where_clause.clone();
    let inputs = sig.inputs.clone();
    let arguments = {
        let mut list = vec![];
        for x in sig.inputs.iter() {
            if let syn::FnArg::Typed(y) = x {
                if let syn::Pat::Ident(ident) = *y.pat.clone() {
                    list.push(ident);
                } else {
                    panic!("patterns on parameters are not supported")
                }
            } else {
                panic!("receiver parameters are not supported")
            }
        }
        list
    };
    if sig.variadic.is_some() {
        panic!("variadic parameters are not supported");
    }
    let output = sig.output.clone();
    let mut versions = quote::quote! {};
    let mut branches = quote::quote! {};
    for version in attr.versions {
        let target = version.target.clone();
        let name = syn::Ident::new(
            &format!("{name}_{}", target.replace(":", "_").replace(".", "_")),
            proc_macro2::Span::mixed_site(),
        );
        let s = target.split(":").collect::<Vec<&str>>();
        let target_cpu = target::TARGET_CPUS
            .iter()
            .find(|target_cpu| target_cpu.target_cpu == s[0])
            .expect("unknown target_cpu");
        let additional_target_features = s[1..].to_vec();
        let target_arch = target_cpu.target_arch;
        let target_cpu = target_cpu.target_cpu;
        if !version.import {
            versions.extend(quote::quote! {
                #[inline]
                #[cfg(any(target_arch = #target_arch))]
                #[crate::target_cpu(enable = #target_cpu)]
                #(#[target_feature(enable = #additional_target_features)])*
                fn #name < #generics_params > (#inputs) #output #generics_where { #block }
            });
        }
        branches.extend(quote::quote! {
            #[cfg(target_arch = #target_arch)]
            if crate::is_cpu_detected!(#target_cpu) #(&& crate::is_feature_detected!(#additional_target_features))* {
                let _multiversion_internal: unsafe fn(#inputs) #output = #name;
                CACHE.store(_multiversion_internal as *mut (), core::sync::atomic::Ordering::Relaxed);
                return unsafe { _multiversion_internal(#(#arguments,)*) };
            }
        });
    }
    quote::quote! {
        #versions
        fn fallback < #generics_params > (#inputs) #output #generics_where { #block }
        #[inline(always)]
        #(#attrs)* #vis #sig {
            static CACHE: core::sync::atomic::AtomicPtr<()> = core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());
            let cache = CACHE.load(core::sync::atomic::Ordering::Relaxed);
            if !cache.is_null() {
                let f = unsafe { core::mem::transmute::<*mut (), unsafe fn(#inputs) #output>(cache as _) };
                return unsafe { f(#(#arguments,)*) };
            }
            #branches
            let _multiversion_internal: unsafe fn(#inputs) #output = fallback;
            CACHE.store(_multiversion_internal as *mut (), core::sync::atomic::Ordering::Relaxed);
            unsafe { _multiversion_internal(#(#arguments,)*) }
        }
    }
    .into()
}

struct TargetCpu {
    enable: String,
}

impl syn::parse::Parse for TargetCpu {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _: syn::Ident = input.parse()?;
        let _: syn::Token![=] = input.parse()?;
        let enable: syn::LitStr = input.parse()?;
        Ok(Self {
            enable: enable.value(),
        })
    }
}

#[proc_macro_attribute]
pub fn target_cpu(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let attr = syn::parse_macro_input!(attr as TargetCpu);
    let mut result = quote::quote! {};
    for s in attr.enable.split(',') {
        let target_cpu = target::TARGET_CPUS
            .iter()
            .find(|target_cpu| target_cpu.target_cpu == s)
            .expect("unknown target_cpu");
        let target_features = target_cpu.target_features;
        result.extend(quote::quote!(
            #(#[target_feature(enable = #target_features)])*
        ));
    }
    result.extend(proc_macro2::TokenStream::from(item));
    result.into()
}

#[proc_macro]
pub fn define_is_cpu_detected(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let target_arch = syn::parse_macro_input!(input as syn::LitStr).value();
    let mut arms = quote::quote! {};
    for target_cpu in target::TARGET_CPUS {
        if target_cpu.target_arch != target_arch {
            continue;
        }
        let target_cpu = target_cpu.target_cpu;
        let ident = syn::Ident::new(
            &format!("is_{}_detected", target_cpu.replace('.', "_")),
            proc_macro2::Span::mixed_site(),
        );
        arms.extend(quote::quote! {
            (#target_cpu) => { $crate::internal::#ident() };
        });
    }
    let ident = syn::Ident::new(
        &format!("is_{target_arch}_cpu_detected"),
        proc_macro2::Span::mixed_site(),
    );
    quote::quote! {
        #[macro_export]
        macro_rules! #ident {
            #arms
        }
    }
    .into()
}
