import pypandoc

if __name__ == '__main__':
    # This script converts the README.md file from markdown format to README.rst, formatted in reStructuredText
    # Taken from https://tom-christie.github.io/articles/pypi/ (November 9, 2016)

    # converts markdown to reStructured
    z = pypandoc.convert('../README.md', 'rst', format='markdown')

    # writes converted file
    with open('./README.rst', 'w') as outfile:
        outfile.write(z)
