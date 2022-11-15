check = "AbC"
result=""
for i in check:
    if(i.isupper()):
        result+="@"
    if(i.islower()):
        result+="$"

print(result)        