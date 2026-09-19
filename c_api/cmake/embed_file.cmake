# Concatenates text files into a C header defining <SYMBOL> as a string
# literal, so shader sources ship inside the library and are compiled at
# runtime.
#
# Usage (script mode):
#   cmake -DINPUTS=<file1;file2;...> -DOUTPUT=<header> -DSYMBOL=<name> -P embed_file.cmake

# The shader sources #include each other for the offline (tensor) build; the
# runtime compiler has no include path, so for the concatenated source those
# lines are dropped (every file carries an include guard, order is fixed by
# the caller).
set(content "")
foreach(input IN LISTS INPUTS)
    file(READ "${input}" part)
    string(REGEX REPLACE "#include \"anbc_[a-z0-9_]+\\.metal\"" "" part "${part}")
    string(APPEND content "${part}\n")
endforeach()

# A raw string literal keeps the source verbatim; the delimiter just has to
# not appear in the shader. Very large literals are split into chunks because
# some compilers cap a single raw string.
set(delim "anbc_src")
set(header "/* Generated from ${INPUTS} -- do not edit. */\n#pragma once\nstatic const char ${SYMBOL}[] =\n")
string(LENGTH "${content}" total)
set(pos 0)
set(chunk 60000)
while(pos LESS total)
    string(SUBSTRING "${content}" ${pos} ${chunk} piece)
    string(APPEND header "R\"${delim}(${piece})${delim}\"\n")
    math(EXPR pos "${pos} + ${chunk}")
endwhile()
string(APPEND header ";\n")
file(WRITE "${OUTPUT}" "${header}")
