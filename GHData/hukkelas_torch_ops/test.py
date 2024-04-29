from typing import Any
class A:

    def __init__(self) -> None:
        self.A = 1

    def __getattr__(self, __name: str) -> Any:
        print(__name)


print(A().A)
print(A().B)