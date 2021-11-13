from inspect import signature
from functools import wraps


def typeassert(*ty_args, **ty_kwargs):
    # ref:https://python3-cookbook.readthedocs.io/zh_CN/latest/c09/p07_enforcing_type_check_on_function_using_decorator.html
    def decorate(func):
        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError(
                            f'Argument {name} must be {bound_types[name]}'
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorate
