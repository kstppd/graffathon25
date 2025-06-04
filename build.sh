BASE=$(pwd)
RAYLIB_GIT="https://github.com/raysan5/raylib.git"
INC_FLAGS="-I./raylib/src" 
LD_FLAGS=" -Wl,--gc-sections  -fno-use-cxa-atexit -nostartfiles -no-pie -s -L./raylib/src  -Wl,-rpath=./raylib/src -fopenmp -lraylib"
CXX_FLAGS="-s -no-pie -fno-use-cxa-atexit  -fdata-sections -ffunction-sections -fno-rtti -fno-exceptions   -Wall -Wextra -Werror -std=c++20 -Os"

git clone ${RAYLIB_GIT} --depth=1 -b master
cd raylib/src
make PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=SHARED -j 12
cd ${BASE}

g++ -c  main.cpp ${INC_FLAGS} ${CXX_FLAGS}  -fopenmp
nasm -f elf64 start.asm 
g++ start.o  main.o   ${LD_FLAGS}  -lraylib -o demo
rm start.o main.o
ls -al demo
ldd demo
