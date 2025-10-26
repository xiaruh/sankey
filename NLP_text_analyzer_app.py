"""
Group members: Matthew Xue, Josue Ramirez Antonio, Ruhan Xia
filename: NLP_text_analyzer_app.py
description: Application of reusable library for Natural Language Processing
"""

import NLP_text_analyzer_lib as NLP
import os
import pprint as pp # Pretty printer

def main():

    # Create NLP objects
    authors = NLP.NLPTextAnalyzer()
    letters = NLP.NLPTextAnalyzer()

    # Define root directories for text files
    letters_root_directory = 'data/Individual Letters'
    grouped_letters_root_directory = 'data/Combined Letters'
    letters_paths = letters.get_text_paths(letters_root_directory)
    combined_paths = authors.get_text_paths(grouped_letters_root_directory)

    for path in letters_paths:
        try:
            # Splitting path to get labels
            label_parts = path.split(os.sep)
            label = label_parts[-2] + ' ' + os.path.splitext(label_parts[-1])[0]
            letters.load_text(path, label=label)
        except Exception as e:
            print(str(e))

    for path in combined_paths:
        try:
            # Using basename to get the file name without extension
            label = os.path.splitext(os.path.basename(path))[0]
            authors.load_text(path, label=label)
        except Exception as e:
            print(str(e))

    # Process the texts and get results
    result = authors.load_all_text(letters_paths)

    # Plot the Sankey diagram
    authors.plot_sankey(result)

    # # Combine .txt letters example usage:
    # # folder_path = r"C:\Users\hafid\OneDrive\Documents\Classes\ds3500 Advanced Programming with Data\DS3500\HW\HW4\ds3500_hw4\data\Hazard Stevens"
    # # authors.combine_txf_files_and_save(folder_path)

    letters_fig = letters.get_sentiment_plot()
    authors_fig = authors.get_sentiment_plot(isGroupedAuthor=True)
    authors.plot_sentiment(letters_fig, authors_fig)
    #
    letters.generate_wordclouds_subplots()


if __name__ == '__main__':
    main()
