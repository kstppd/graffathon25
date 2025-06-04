; start.asm
; Stuff with __dso handle figured out from here https://stackoverflow.com/questions/34308720/where-is-dso-handle-defined + some googling around
; Supossedly this is used to register and use??!?? destructors of global objects such as global vars / static stuff. It seems
; to be used atexit() literally the function to call thos dtors?? 
; x86x64 specific 
BITS 64
EXTERN _exit
EXTERN jump_start
GLOBAL _start

SECTION .data
    GLOBAL __dso_handle
__dso_handle:
    dq 0

SECTION .text

_start:
    call jump_start       ; call jump start from demo code
    mov rdi, 0            ; set rdi to 0 
    call _exit            ; call libc _exit with 0 exit
