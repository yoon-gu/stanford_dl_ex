function retval = isOctave()
    retval = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end