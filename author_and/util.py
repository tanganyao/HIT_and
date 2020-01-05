
def get_affiliation(affiliation):
    if ';' in affiliation:
        aff = affiliation.split(";")[0]
    elif '.' in affiliation:
        aff = affiliation.split(".")[0]
    elif '\n' in affiliation:
        aff = affiliation.split("\n")[0]
    else:
        aff = ''
    return aff
