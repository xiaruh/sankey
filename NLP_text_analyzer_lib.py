"""
Group members: Matthew Xue, Josue Ramirez Antonio, Ruhan Xia
filename: NLP_text_analyzer_lib.py
description: An extensible reusable library for Natural Language Processing able to
             load and parse .txt and .json files, and create 3 compelling visualizations
             to aid in the study of however many such files a user desires.
"""

from collections import defaultdict, Counter
import os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import sankey as sk
from textblob import TextBlob
import wordcloud as wc
import json
from NLPParserError import NLPParserError


class NLPTextAnalyzer:
    def __init__(self):
        # string  --> {filename/label --> statistics}, e.g.,
        # "wordcounts" --> {"A": wc_A, "B": wc_B, ....}
        self.data = defaultdict(dict)
        self.wordcount_df = pd.DataFrame()
        self.paths = list()

    def _save_results(self, label, results):
        """
        Save parsed file statistics in the internal data dictionary
        """
        for k, v in results.items():
            self.data[k][label] = v

    @NLPParserError.check_file_format
    def _default_parser(self, filepath, stop_words):
        """
        Default parser that removes common punctuations, turns all text to lowercase, removes any numbers
        in text, removes common stopwords, and computes statistics such as words count, num. words, etc
        """
        PUNC = [" ", "(", ")", '"', '?', '[', ']', '', '.', '&', '\\', '\\\\', ',', '//', '//',
                '\\', '\\\\', '!']
        text = []
        all_text = ''

        # Read data from text files
        with open(filepath, 'r') as f:
            for line in f.readlines():
                text.append(line.strip())

        # Remove Punctuation
        for i in range(len(text)):
            for ele in text[i]:
                if ele in PUNC:
                    text[i] = text[i].replace(ele, ",")
            all_text += text[i].lower()

        word_list = all_text.split(',')

        # Remove numbers
        for i in range(len(word_list)):
            word_list[i] = ''.join([j for j in word_list[i] if not j.isdigit()])

        # Remove empty characters
        while '' in word_list:
            word_list.remove('')

        # Remove stopwords
        stopwords = self.load_stop_words(stopfile=stop_words)
        for w in stopwords:
            while w in word_list:
                word_list.remove(w)

        # Join words list elts to string
        words_string = ' '.join(word_list)

        # Instantiate TextBlob object with words string to get sentiment scores
        blob = TextBlob(words_string)

        # Construct results dict
        results = {'wordcount': Counter(word_list),
                   'numwords': len(word_list),
                   'polarity': blob.sentiment.polarity,
                   'subjectivity': blob.sentiment.subjectivity,
                   'allwords': words_string}
        return results

    @NLPParserError.check_file_format
    def json_parser(self, filepath, stop_words):
        """
        Simple json parser that turns all text to lowercase, removes any numbers in text,
        removes common stopwords, and computes statistics such as words count, num. words, etc
        """
        f = open(filepath)
        raw = json.load(f)
        text = raw['text']
        words = text.split(" ")

        # Remove stopwords
        stopwords = self.load_stop_words(stopfile=stop_words)
        for w in stopwords:
            while w in words:
                words.remove(w)

        # Remove numbers
        for i in range(len(words)):
            words[i] = ''.join([j for j in words[i] if not j.isdigit()])

        # Join words list elts to string
        words_string = ' '.join(words)

        # Instantiate TextBlob object with words string to get sentiment scores
        blob = TextBlob(words_string)

        # Construct results dict
        results = {'wordcount': Counter(words),
                   'numwords': len(words),
                   'polarity': blob.sentiment.polarity,
                   'subjectivity': blob.sentiment.subjectivity,
                   'allwords': words_string}
        f.close()
        return results

    def load_text(self, filename, label=None, parser=None):
        """ Registers a text document with the framework
        Extracts and stores data to be used in later
        visualizations. """

        if parser is None:
            results = self._default_parser(filename, 'data/stopwords.txt')
        else:
            results = parser(filename, 'data/stopwords.txt')

        if label is None:
            label = filename

        # store the results of processing one file
        # in the internal state (data)
        self._save_results(label, results)

    @staticmethod
    def load_stop_words(stopfile=None):
        """ Load stop words"""
        if stopfile:
            with open(stopfile, 'r') as f:
                stopwords = f.read().splitlines()
        return stopwords

    def load_all_text(self, filepaths, parser=None):
        """ Load and combine texts from multiple files, organizing by author and text name
        """
        all_results = {'wordcount': {}}

        for filepath in filepaths:

            # Split the file path to get the author and text name
            parts = filepath.split(os.sep)

            # NEED TO USE IF STATEMENTS TO MAKE SURE CODE RUNS FOR BOTH WINDOWS AND MAC USERS
            # FOR MAC USERS PARTS WILL HAVE 4 ELTS AND FOR WINDOWS USERS IT WILL HAVE 3 ELTS
            if len(parts) == 4:
                author = parts[2]
                text_name = parts[3].split('.')[0]
                author_text_key = (author, text_name)

                if author_text_key not in all_results['wordcount']:
                    all_results['wordcount'][author_text_key] = Counter()

                if parser is None:
                    results = self._default_parser(filepath, 'data/stopwords.txt')
                else:
                    results = parser(filepath, 'data/stopwords.txt')

                # Combine word counts for this author and text
                all_results['wordcount'][author_text_key].update(results['wordcount'])

            elif len(parts) == 3:
                author = parts[1]
                text_name = parts[2].split('.')[0]
                author_text_key = (author, text_name)

                if author_text_key not in all_results['wordcount']:
                    all_results['wordcount'][author_text_key] = Counter()

                if parser is None:
                    results = self._default_parser(filepath, 'data/stopwords.txt')
                else:
                    results = parser(filepath, 'data/stopwords.txt')

                # Combine word counts for this author and text
                all_results['wordcount'][author_text_key].update(results['wordcount'])
        return all_results


    @staticmethod
    def flatten_wordcount_to_dataframe(all_results):
        """
        Flattens nested word count data for each author and text, and converts it to a DataFrame.
        Args:
            all_results: a dictionary of word counts for each author and text.
        Returns:
            DataFrame with columns named 'Author', 'TextDate', 'Word', and 'Count'.
        """
        flattened_data = []
        for (author, text_date), counter in all_results['wordcount'].items():
            for word, count in counter.items():
                flattened_data.append((author, text_date, word, count))

        df = pd.DataFrame(flattened_data, columns=['Author', 'TextDate', 'Word', 'Count'])
        return df

    def plot_sankey(self, all_results):
        """
        Plots a multi-layered Sankey diagram based on word counts for top 5 words of each text file for each author.
        Args:
            all_results: a dictionary of word counts for each author and text.
        """
        # Get DataFrame from the combined function
        df = self.flatten_wordcount_to_dataframe(all_results)

        # Group by Author and TextDate, then get the top 5 words for each group
        top_words_per_group = df.groupby(['Author', 'TextDate']).apply(
            lambda x: x.sort_values(by='Count', ascending=False).head(3)
        ).reset_index(drop=True)

        # Plot the Sankey diagram with the top words per group
        sk.make_sankey(top_words_per_group, 'Author', 'TextDate', 'Word', vals='Count')

    def generate_wordclouds_subplots(self, rows=3, cols=4):
        """
        Generates a subplot of Word Clouds for each individual text file uploaded by the user.
        Args:
            rows: the amount of rows the user wants for the subplot
            cols: the amount of columns that user wants for the subplot

        Returns:
            None
        """
        # Retrieve the 'allwords' from the results data
        cloud = wc.WordCloud(colormap='Reds', background_color='black')

        # Calculate subplot layout based on the number of entries in allwords
        num_entries = len(self.data['allwords'])

        fig, axes = plt.subplots(rows, cols, figsize=(30, 15))  # Increase the figsize here

        fig.tight_layout(pad=3.0)  # Adjust spacing between subplots
        
        for idx, (author, words) in enumerate(self.data['allwords'].items()):
            row = idx // cols
            col = idx % cols

            # Generate word cloud for the given author
            letter_cloud = cloud.generate(words)

            # Plot on the respective subplot
            if num_entries > 1:
                if rows == 1:
                    axes[col].imshow(letter_cloud)
                    axes[col].set_title(author, color="red")
                    axes[col].axis('off')
                else:
                    axes[row, col].imshow(letter_cloud)
                    axes[row, col].set_title(author, color="red")
                    axes[row, col].axis('off')
            else:  # For a single entry, don't use subplots
                plt.imshow(letter_cloud)
                plt.title(author, color="red")
                plt.axis('off')
         
        # Hide empty subplots (if any)
        for idx in range(num_entries, rows * cols):
            if num_entries == 1:  # For single entry, axes is not an array
                axes.axis('off')
            else:
                row = idx // cols
                col = idx % cols
                axes[row, col].axis('off')
                
        plt.show()

    def get_sentiment_plot(self, isGroupedAuthor=False):
        """
        Get polarity vs subjectivity scatter plot as figure object without displaying it
        Args: boolean to differentiate whether loaded text is not grouped by author (files will
              have dates) from text that's grouped by author (files will not have dates)
        """
        if len(self.data) != 0:
            rows = []
            if not isGroupedAuthor:
                for author_date, values in self.data['polarity'].items():

                    # Index where 18th century year ('18**') starts
                    year_start = author_date.find(' 18')

                    # Get text author and date from index found above
                    author = author_date[:year_start+1]
                    date = author_date[year_start:]

                    # Create one row of df which will be used to plot sentiment scores
                    row = {'Author': author, 'Date': date, 'Polarity': values,
                           'Subjectivity': self.data['subjectivity'][author_date]}
                    rows.append(row)

                # Create a DataFrame from the list of dictionaries
                df = pd.DataFrame(rows)

                # Create a scatter plot with color by Author
                fig = px.scatter(df, x='Polarity', y='Subjectivity', color='Author', hover_data=['Date'])
                fig.update_traces(marker=dict(size=8,
                                              line=dict(width=1,
                                                        color='DarkSlateGrey')),
                                  selector=dict(mode='markers'))
                return fig
            else:
                for author, values in self.data['polarity'].items():
                    # Create one row of df which will be used to plot sentiment scores
                    row = {'Author': author, 'Polarity': values,
                           'Subjectivity': self.data['subjectivity'][author]}
                    rows.append(row)

                # Create a DataFrame from the list of dictionaries
                df = pd.DataFrame(rows)

                # Create a scatter plot with color by Author
                fig = px.scatter(df, x='Polarity', y='Subjectivity', color='Author')
                fig.update_traces(marker=dict(size=18,
                                              line=dict(width=2,
                                                        color='DarkSlateGrey')),
                                  selector=dict(mode='markers'))
                return fig
        else:
            # Return empty scatter object if NLP instance doesn't have any loaded texts
            return px.scatter()

    @staticmethod
    def plot_sentiment(fig1, fig2):
        """
        Overlay two scatter figures and display combined sentiment plot
        Aegs: 2 scatter figure objects
        """

        # Create a new figure to combine the two scatters
        combined_fig = px.scatter()

        # Add traces from fig1 to the combined_fig
        for trace in fig1['data']:
            combined_fig.add_trace(trace)

        # Add traces from fig2 to the combined_fig
        for trace in fig2['data']:
            combined_fig.add_trace(trace)

        combined_fig.update_layout(
            title='Sentiment Analysis',
            title_x=0.45,
            xaxis_title='Polarity',
            yaxis_title='Subjectivity'
        )

        # Show the combined scatter plot
        combined_fig.show()

    @staticmethod
    def combine_txf_files_and_save(folder_path):
        """
        Read all .txt files in given directory and combine them into a new .txt file
        """
        output_file = os.path.join(folder_path, 'combined_output.txt')

        with open(output_file, 'w', encoding='utf-8') as combined_file:
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as txf_file:
                        content = txf_file.read()
                        combined_file.write(content + '\n')

    @staticmethod
    def get_text_paths(root_directory):
        """
        Retrieve all .txt file paths in give directory
        """
        paths = []

        # Iterate over subdirectories in the root directory
        for author_dir in os.listdir(root_directory):
            author_path = os.path.join(root_directory, author_dir)

            # Check if it's a directory and not a file
            if os.path.isdir(author_path):
                # Iterate over files in the author's directory
                for filename in os.listdir(author_path):
                    # Avoid reading invisible files and non-txt files
                    if not filename.startswith('.') and filename.endswith('.txt'):
                        file_path = os.path.join(author_path, filename)
                        paths.append(file_path)
        return paths
