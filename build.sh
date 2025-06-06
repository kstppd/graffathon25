BASE=$(pwd)
RAYLIB_GIT="https://github.com/raysan5/raylib.git"
INC_FLAGS="-isystem./raylib/src" 
LD_FLAGS=" -Wl,--gc-sections  -fno-use-cxa-atexit -nostartfiles -no-pie -s -L./raylib/src  -Wl,-rpath=./raylib/src  -lraylib"
CXX_FLAGS="-s -no-pie  -fno-use-cxa-atexit  -fdata-sections -ffunction-sections -fno-rtti -fno-exceptions   -Wall -Wextra  -std=c++20 -Os"

git clone ${RAYLIB_GIT} --depth=1 -b master
cd raylib/src
make PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=SHARED -j 12
clear
cd ${BASE}

EXTRA=" -nostdlib -fopenmp -fno-unroll-loops -Wl,--build-id=none -Wl,-z,norelro -ffast-math -fsingle-precision-constant -fmerge-all-constants -fno-unroll-loops -fno-math-errno -falign-functions=1 -falign-jumps=1 -falign-loops=1 -fno-stack-protector -fomit-frame-pointer -ffunction-sections -fdata-sections -Wl,--gc-sections"
g++ -c ${EXTRA} main.cpp ${INC_FLAGS} ${CXX_FLAGS}
nasm -f elf64 start.asm -Ox
g++  ${EXTRA}   start.o  main.o  ${LD_FLAGS} -lc -lm -lraylib -o demo
strip -S --strip-unneeded --remove-section=.note.gnu.gold-version --remove-section=.comment --remove-section=.note --remove-section=.note.gnu.build-id --remove-section=.note.ABI-tag demo
rm start.o main.o
ls -ahl demo
