
function anneal_linear(pctg, from, to)
    return (1 - pctg) * from + pctg * to
end

function anneal_cosine(pctg, from, to)
    return to + (from - to)/2 * (cos(pi*pctg) + 1)
end

anneal_const(pctg, from, to) = from

anneal_exp(pctg, from, to) = from * (to/from)^pctg
