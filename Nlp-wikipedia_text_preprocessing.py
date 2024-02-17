#################################################
# TASKS:
#################################################

# Task 1: Write a function to perform text preprocessing operations on the text.
# •	Perform case folding (convert text to lowercase).
# •	Remove punctuation marks.
# •	Remove numeric expressions.


def clean_text(text):
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.apply(lambda x: re.sub("[^\w\s]", "", str(x)))
    text = text.str.replace("\n" , '')
    # Numbers
    text = text.apply(lambda x: re.sub("[\d]", "", x))
    return text

df["text"] = clean_text(df["text"])


def remove_stopwords(text):
    stop_words = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

df["text"] = remove_stopwords(df["text"])


sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))


df["text"].apply(lambda x: TextBlob(x).words)


# ran, runs, running -> run (normalization)

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.head()



tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index() # will need to update the code

tf.head()



tf.columns = ["words", "tf"]
# Visualization of words occurring more than 5000 times
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()



# merging words
text = " ".join(i for i in df["text"])

# defining properties for wordcloud visualization
wordcloud = WordCloud(max_font_size=50,
max_words=100,
background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


df = pd.read_csv("Module_8_Natural_Language_Processing/datasets/wiki_data.csv", index_col=0)


def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
    Performs preprocessing operations on texts.

    :param text: The variable containing texts in the DataFrame
    :param Barplot: Barplot visualization
    :param Wordcloud: Wordcloud visualization
    :return: text


    Example:
            wiki_preprocess(dataframe[col_name])

    """
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace("\n", '')
    # Numbers
    text = text.str.replace('\d', '')
    # Stopwords
    sw = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))


    if Barplot:
        # Term Frequency Calculation
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # Column Naming
        tf.columns = ["words", "tf"]
        # Visualization of words occurring more than 5000 times
        tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
        plt.show()

    if Wordcloud:
        # merging words
        text = " ".join(i for i in text)
        # defining properties for wordcloud visualization
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text

wiki_preprocess(df["text"])

wiki_preprocess(df["text"], True, True)
