

print('item {')
print('\tname: "none_of_the_above"')
print('\tlabel: 0')
print('\tdisplay_name: "background"')
print('}')

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

for i in range(1, 27):
    print('item {')
    print('\tname: "' + letters[i-1].upper() + '"')
    print('\tlabel: ' + str(i))
    print('\tdisplay_name: "' + letters[i-1].upper() + '"')
    print('}')