#!/usr/bin/env python3.8

def remove_content_after_pattern(file_path, save_path='/home/lisha/sdav_integration/smartdata/logs/'):
    with open(file_path, 'r') as file:
        content = file.read()

    # Find the first occurrence of '.-9'
    index_9 = content.find('.-9')

    if index_9 != -1:
        # Find the last occurrence of '(u={' before '.-9'
        index_u = content.rfind('(u={', 0, index_9)

        if index_u != -1:
            # Remove content from the last occurrence of '(u={' onwards
            cleaned_content = content[:index_u]
        else:
            cleaned_content = content  # No '(u={' found before '.-9'
    else:
        cleaned_content = content  # No '.-9' found

    # Write the cleaned content back to the file
    with open(save_path+'clean_sniffer.log', 'w') as file:
        file.write(cleaned_content)

    print("mistakes cleared")
    return cleaned_content

if __name__ == "__main__":
    # Example usage
    file_path = '/home/lisha/sdav_integration/smartdata/logs/sniffer.log'  # Replace with your file path
    remove_content_after_pattern(file_path)

