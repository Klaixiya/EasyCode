module To

export @to

function single_replace(arg::Any, dict, dict_keys)
    return arg
end

function single_replace(arg::Symbol, dict, dict_keys)
    if arg in dict_keys
        return dict[arg]
    else
        return arg
    end
end

function replace_recursion!(arg, dict, dict_keys)
    if !(:args in propertynames(arg))
        return single_replace(arg, dict, dict_keys)
    end
    arg_length = length(arg.args)
    for i in 1:arg_length
        arg.args[i] = replace_recursion!(arg.args[i], dict, dict_keys)
    end
    return arg
end



function to(expr, dict_tuples)
    dict_length = length(dict_tuples.args)
    dict = Dict()
    for i in 1:dict_length
        push!(dict, dict_tuples.args[i].args[2] => dict_tuples.args[i].args[3])
    end
    dict_keys = keys(dict)
    
    replace_recursion!(expr, dict, dict_keys)

    return expr

end

macro to(expr, dict_tuples)
    esc(to(expr, dict_tuples))
end
    
end
