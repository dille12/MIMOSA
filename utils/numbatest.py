from numba import jit
import threading
import time

def notify_first_compile(app):
    def decorator(func):
        has_compiled = set()
        lock = threading.Lock()

        def wrapper(*args, **kwargs):
            key = tuple(type(arg) for arg in args)

            with lock:
                if key not in has_compiled:
                    if key not in func.overloads:
                        app.notify("🔧 Compiling function for the first time...")
                    has_compiled.add(key)

            return func(*args, **kwargs)

        return wrapper
    return decorator


class DummyApp:
    def notify(self, message):
        print(message)
app = DummyApp()

@notify_first_compile(app=app)
@jit(nopython=True)
def my_function(x):
    return x * x



if __name__ == "__main__":
    # Example usage
    for i in range(1000):
        print(my_function(i))
    time.sleep(1)  # Sleep for a second to allow the compilation message to be printed
    print(my_function(20))
