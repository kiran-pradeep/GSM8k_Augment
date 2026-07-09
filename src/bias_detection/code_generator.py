"""
WARNING: Threads can't be forcibly killed mid-execution (unlike processes), so 
a truly infinite loop will hang until the pool worker exits. 
For math code generation this is rarely a problem.
"""

import re
import queue
import threading

def extract_python_code(text: str) -> str:
    text = text.strip()
    fence_pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        text = match.group(1)
    text = text.strip()
    start = text.find("def solve")
    if start == -1:
        raise ValueError("No 'def solve' function found in generated code.")
    return text[start:].strip()


def _execute_code(code: str, result_queue: queue.Queue):
    local_namespace = {}
    try:
        # Pass local_namespace as globals AND locals so that
        # get_answer() can see solve() when it calls it.
        exec(code, local_namespace)   # <-- only one dict, used as globals
        if "get_answer" not in local_namespace:
            result_queue.put(("error", "get_answer() not found"))
            return
        result = local_namespace["get_answer"]()
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))

def run_generated_code(python_code: str, timeout: int = 10):
    result_queue = queue.Queue()
    thread = threading.Thread(
        target=_execute_code,
        args=(python_code, result_queue),
        daemon=True  # Thread (not process) — safe inside pool workers
    )
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Thread can't be forcibly killed, but daemon=True means it won't
        # block process exit. Raise timeout to the caller.
        raise TimeoutError(f"Execution exceeded {timeout} seconds")

    if result_queue.empty():
        raise RuntimeError("No result returned from execution")

    status, payload = result_queue.get()
    if status == "error":
        raise RuntimeError(f"Execution failed: {payload}")

    return payload



# import multiprocessing as mp
# import re

# def extract_python_code(text: str) -> str:
#     text = text.strip()

#     # Remove markdown fences
#     fence_pattern = r"```(?:python)?\n(.*?)```"
#     match = re.search(fence_pattern, text, re.DOTALL)
#     if match:
#         text = match.group(1)

#     text = text.strip()

#     # Trim anything before first function
#     start = text.find("def solve")
#     if start == -1:
#         raise ValueError("No 'def solve' function found in generated code.")
#     text = text[start:]

#     return text.strip()


# def _execute_code(code: str, queue: mp.Queue):
#     local_namespace = {}

#     try:
#         exec(code, {}, local_namespace)

#         if "get_answer" not in local_namespace:
#             queue.put(("error", "get_answer() not found"))
#             return

#         result = local_namespace["get_answer"]()
#         queue.put(("success", result))

#     except Exception as e:
#         queue.put(("error", str(e)))


# def run_generated_code(python_code: str, timeout: int = 10):
#     queue = mp.Queue()
#     process = mp.Process(target=_execute_code, args=(python_code, queue))

#     process.start()
#     process.join(timeout)

#     if process.is_alive():
#         process.terminate()
#         process.join()
#         raise TimeoutError(f"Execution exceeded {timeout} seconds")

#     if queue.empty():
#         raise RuntimeError("No result returned from execution")

#     status, payload = queue.get()

#     if status == "error":
#         raise RuntimeError(f"Execution failed: {payload}")

#     return payload