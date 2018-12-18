import time
import sys
# for x in range(0, 5):
#     b = "Loading" + "." * x
#     print(b, end="\r")
#     time.sleep(1)

for x in range(50):
    b = "Loading" + "." * (x % 5)
    sys.stdout.write(b)
    sys.stdout.flush()
    # sys.stdout.write("\033[F")
    time.sleep(0.5)
