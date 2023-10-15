docker build -t task1 .
#docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task1
docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --security-opt apparmor=unconfined task1
#docker run --rm -it -v "$( cd "$( dirname "$0" )" && pwd )":/results task1 /bin/bash

