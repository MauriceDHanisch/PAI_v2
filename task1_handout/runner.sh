docker build --tag task1 .
#docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task1
docker run --rm -it -v "$( cd "$( dirname "$0" )" && pwd )":/results task1 /bin/bash

