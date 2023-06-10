# Generate requirements.txt
requirements = [
    'torch',
    'torchvision'
]

with open('requirements.txt', 'w') as f:
    for requirement in requirements:
        f.write(requirement + '\n')

print('requirements.txt generated.')