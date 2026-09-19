# Turns a binary file into a C header defining <SYMBOL>[] (unsigned char) and
# <SYMBOL>_size, so a precompiled .metallib can ship inside the library.
#
#   cmake -DINPUT=<file> -DOUTPUT=<header> -DSYMBOL=<name> -P embed_binary.cmake

file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hexlen)
math(EXPR size "${hexlen} / 2")
# "0a1b2c" -> "0x0a,0x1b,0x2c," with a newline every 32 bytes
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){32})" "\\1\n" bytes "${bytes}")
file(WRITE "${OUTPUT}"
"/* Generated from ${INPUT} -- do not edit. */
#pragma once
static const unsigned char ${SYMBOL}[] = {
${bytes}
};
static const unsigned long ${SYMBOL}_size = ${size};
")
