; Here we provide handles to start and exit as wel
; as jump_start becasue in our demo we have nuked main()
; to save space
format ELF64 ; one machine only

section '.data' writeable
public __dso_handle
__dso_handle:
    dq 0

section '.text' executable
public _start
extrn _exit
extrn jump_start

_start:
    call jump_start
    mov rdi, 0
    call _exit

