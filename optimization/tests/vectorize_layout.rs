use optimization::{Vectorize, flat_view, flatten_value, symbolic_value, unflatten_value};
use sx_core::SX;

#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Xyz<T> {
    x: T,
    y: T,
    z: T,
}

#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Ab<T> {
    a: T,
    b: Xyz<T>,
}

#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct WithArray<T> {
    head: T,
    tail: [Xyz<T>; 2],
}

#[test]
fn flatten_unflatten_roundtrips_simple_nested_types() {
    let value = Ab {
        a: 1.5,
        b: Xyz {
            x: -2.0,
            y: 3.25,
            z: 4.5,
        },
    };
    let flat = flatten_value(&value);
    assert_eq!(flat, vec![1.5, -2.0, 3.25, 4.5]);

    let rebuilt = unflatten_value::<Ab<f64>, f64>(&flat).expect("layout should unflatten");
    assert_eq!(rebuilt, value);
}

#[test]
fn flatten_unflatten_roundtrips_arrays_of_nested_types() {
    let value = WithArray {
        head: 9.0,
        tail: [
            Xyz {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            Xyz {
                x: 4.0,
                y: 5.0,
                z: 6.0,
            },
        ],
    };
    let flat = flatten_value(&value);
    assert_eq!(flat, vec![9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let rebuilt =
        unflatten_value::<WithArray<f64>, f64>(&flat).expect("array layout should unflatten");
    assert_eq!(rebuilt, value);
}

#[test]
fn generated_borrowed_view_is_correct_for_flat_numeric_slices() {
    let flat = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let view: WithArrayView<'_, f64> =
        flat_view::<WithArray<f64>, f64>(&flat).expect("flat slice should project into view");

    assert_eq!(*view.head, 10.0);
    assert_eq!(*view.tail[0].x, 11.0);
    assert_eq!(*view.tail[0].y, 12.0);
    assert_eq!(*view.tail[0].z, 13.0);
    assert_eq!(*view.tail[1].x, 14.0);
    assert_eq!(*view.tail[1].y, 15.0);
    assert_eq!(*view.tail[1].z, 16.0);
}

#[test]
fn generated_borrowed_view_is_correct_for_owned_structs() {
    let value = Ab {
        a: 1.0,
        b: Xyz {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        },
    };
    let view: AbView<'_, f64> = value.view();

    assert_eq!(*view.a, 1.0);
    assert_eq!(*view.b.x, 2.0);
    assert_eq!(*view.b.y, 3.0);
    assert_eq!(*view.b.z, 4.0);
}

#[test]
fn symbolic_layout_uses_field_order_as_flatten_order() {
    let symbolic = symbolic_value::<WithArray<SX>>("state").expect("symbolic layout should build");
    let names = symbolic
        .flatten_cloned()
        .into_iter()
        .map(|sx| sx.to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec![
            "state_0", "state_1", "state_2", "state_3", "state_4", "state_5", "state_6",
        ]
    );
}
