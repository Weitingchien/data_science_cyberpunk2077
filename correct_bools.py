import json


def convert_bools(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = convert_bools(value)
        elif isinstance(value, bool):
            d[key] = True if value else False
    # print(f'd: {d}')
    return d



def main(filename, output_filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        result = convert_bools(item)
    print(f'result: {result}')

    with open(output_filename, 'w', encoding='utf-8') as file_output:
        json.dump(result, file_output, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    filename = 'reviews.json'
    output_filename = 'modified.json'
    main(filename, output_filename)