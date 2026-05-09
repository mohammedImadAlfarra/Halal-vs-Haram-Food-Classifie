{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO4+wKIRqWRMfnqSOW0z5mV",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mohammedImadAlfarra/Halal-vs-Haram-Food-Classifie/blob/main/Halal%20vs%20Haram%20Food%20Classifier.PY\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "PROJECT  : Halal vs Haram Food Ingredients Classification\n",
        "\n",
        " DATASET  : Food Ingredients Dataset with Halal Label\n",
        "\n",
        " SOURCE   : Kaggle — @irfanakbarihabibi\n",
        "\n",
        " LINK     : https://www.kaggle.com/datasets/irfanakbarihabibi/\n",
        "            food-ingredients-dataset-with-halal-label\n",
        "\n",
        " FEATURES : Food ingredient text descriptions\n",
        "\n",
        "  TARGET   : Binary label — halal / haram\n",
        "\n",
        "  MODELS   : Naive Bayes  |  Decision Tree\n",
        "  \n",
        " TOTAL    : 39,787 samples  |  2 columns (text, label)"
      ],
      "metadata": {
        "id": "IUfXHjVx772Z"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "pandas (pd) : used for handling structured data (DataFrame like Excel tables)\n",
        "\n",
        "matplotlib.pyplot (plt) : used for creating basic visualizations and plots\n"
      ],
      "metadata": {
        "id": "G5WGJF3bqkNI"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2RgS3XBF7RJK"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score,\n",
        "    classification_report,\n",
        "    confusion_matrix,\n",
        ")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        " LOAD DATASET\n",
        "\n",
        " We load the CSV file from Kaggle into a pandas DataFrame.\n",
        "\n",
        " A DataFrame is like an Excel table that Python can work with.\n"
      ],
      "metadata": {
        "id": "YV90cLQr8nhi"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "print(\"   Loading Dataset\")\n",
        "\n",
        "df = pd.read_csv(\"cleaned_dataset.csv\")\n",
        "\n",
        "\n",
        "print(f\"  Task    : Binary Text Classification\")\n",
        "print(f\"  Classes : halal  |  haram\")\n",
        "print(f\"  Rows    : {df.shape[0]:,}\")\n",
        "print(f\"  Columns : {df.shape[1]}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J1SfV0F688nr",
        "outputId": "3571f21b-0884-4699-fd38-0b9d395d4704"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   Loading Dataset\n",
            "  Task    : Binary Text Classification\n",
            "  Classes : halal  |  haram\n",
            "  Rows    : 39,787\n",
            "  Columns : 2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "DATA DESCRIPTION\n",
        "\n",
        " We explore the dataset to understand what it contains.\n",
        "\n",
        " This helps us choose the right model and preprocessing steps."
      ],
      "metadata": {
        "id": "vVB_s6LE9GXH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "print(\" Data Description & Exploration\")\n",
        "\n",
        "\n",
        "print(\"\\nFirst 5 rows of the dataset:\")\n",
        "print(df.head())\n",
        "\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\nLabel Distribution (Halal vs Haram):\")\n",
        "\n",
        "label_counts = df['label'].value_counts()\n",
        "print(label_counts)\n",
        "print(f\"\\nHalal samples : {label_counts['halal']:,}\")\n",
        "print(f\"Haram samples : {label_counts['haram']:,}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "m1zu4be59O24",
        "outputId": "6bdf417c-30c8-49bc-93b1-b6039bc401c9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " Data Description & Exploration\n",
            "\n",
            "First 5 rows of the dataset:\n",
            "                                                text  label\n",
            "0                                      vegetable oil  halal\n",
            "1  beef stock contains less than of mirepoix carr...  halal\n",
            "2  clam stock potatoes clams cream vegetable oil ...  haram\n",
            "3  water cream broccoli celery vegetable oil corn...  haram\n",
            "4  chicken stock contains less than of yeast extr...  halal\n",
            "======================================================================\n",
            "\n",
            "Label Distribution (Halal vs Haram):\n",
            "label\n",
            "halal    21826\n",
            "haram    17961\n",
            "Name: count, dtype: int64\n",
            "\n",
            "Halal samples : 21,826\n",
            "Haram samples : 17,961\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "VISUALIZATION 1 :\n",
        "\n",
        "Label Distribution (Pie Chart) Shows how balanced the dataset is between halal and haram.\n",
        "\n",
        "A balanced dataset means the model won't be biased.\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "M-mj3NvEB1yZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(6, 6))\n",
        "colors_pie = ['#2ecc71', '#e74c3c']\n",
        "plt.pie(\n",
        "    label_counts.values,\n",
        "    labels=label_counts.index,\n",
        "    autopct='%1.1f%%',\n",
        "    colors=colors_pie,\n",
        "    startangle=90,\n",
        "    textprops={'fontsize': 13}\n",
        ")\n",
        "plt.title('Dataset Label Distribution\\nHalal vs Haram',\n",
        "          fontsize=14, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 607
        },
        "id": "24JngVqVBv41",
        "outputId": "d33443be-c32e-44e5-cc5c-db5c2ad3236c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 1 Axes>"
            ],
            "image/png": "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/091OBwNPicRNR2LEJEbXXoYLSYmBlu3boXZbIYkSU0+3GMymfDTTz85Px46dCiOHz8Om80GSZKqlRJXaNu2La655hoAgCRJ+Oabb6odFrt0NAi48Iv7yy+/RE5ODpYuXYrXX38d9913H0JCQgAABw4cwPTp012asTYbNmyoVrimTZuG8+fPQ5KkRp+h5+fnV+vtixcvdhYShUKBBQsWoLy8HJIkNaqgAqj1cNlFl5cfInI/ftcRuVFBQYHz371793bO26mqqsKyZcuatK+SkhLnZGQAGDVqFBITEwEAhw8fxr59+1yQuLr77rvPOZn3k08+QXZ2NoALv8zHjx9fbdvDhw8jPj4eERERuPnmm523d+7cGU899RQANHvCdFNc+p4DwN13343w8HAAF05Nd9W+g4KCcMcddzhHzVq6byISg0WIyI1SU1Nx9OhRAMBPP/2Ehx56CFFRUfj2229x6NChJu2rTZs2CA4ORnFxMQDg5Zdfxrlz52Cz2fDFF1+4Za2Z0aNHO5/z5MmTzttHjRqFiIiIatu+++67+OqrrzB06FC0b98ekZGRKCwsxJdffuncJjg42OUZL5eamlrt43vuuQdjx45FVlaW80w4V+y7uLgYo0aNQv/+/bFx40asWrWqRfsmIjFYhIjc6JlnnsGKFStgs9ngcDjw2WefAQD8/f0xZswYfP/9943el1qtxvTp052HlwoLC/Haa68BuDDq0r59e+zcudOl+fV6PcaNG4ePP/642u333XdfrdtXVFTUOdKlVCoxbdo0l+arTa9evXDddddhxYoVAICMjAznpO8JEyZg3rx5zd73pEmT8M477+Ds2bMALiyncPF5WrpvIhKDc4SI3GjgwIFYuXIl+vfvD51Oh6CgIFx//fXYvHkzunTp0uT9PfPMM/joo4+QkpICjUaDqKgoPPDAA1i/fj38/f3d8ApqzgWKjIzEyJEja2w3efJkPPPMM7j66qsRHx8PvV4PrVaL+Ph43H777Vi/fj1uueUWt2S83HfffYepU6ciOjoaWq0WHTp0wCuvvOJcHbq5QkNDsXHjRowZMwaBgYEwGAzo06cPvv/+e0ycONE14YmoVSkkqYXr8RMRERF5KY4IERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFssQgRERGRbLEIERERkWyxCBEREZFsqUUHICLvJkkSKhxmmOxmVDossEsOOCDB8dffEiQk5pighAJQKgGFAgqlEtDqoNAboNDrodDpoVAoRL8UIpIhFiEicqqwm3HOWoI8SzHyrCU4ZylBka0cZfYqmOxVKLdXweSogslu/utPFSocZjgg1bvfla/tB6qq6t5AobhQhi4WI73hwh8/f6iCQ6AMDoXy4t8hF/5WBYdAERB0oVQRETUTixCRjJyzlCCzKg+nzQV/FZ4S5FmLkWcpwTlrMcrs9ZQVd5IkSFWVkKoqm/Y4pQrK4BCoIqOhioqBOjr2wr+jY6GKioEqJMw9eYnIZ7AIEfkYSZJw1lKEE1W5OFF1Dicqc3GiKg+ZVXniio67OOxwFObDUZgP68H9Ne5W6A3OYqRu2x7qhCRoEpKgio3nSBIRAQAUkiTVP6ZNRB6r0mFBhukU9ptO4khlDjKr8pBZdQ6VDovoaNU0eGistWl1ULdtD037JKgTLvzRtO8AZUCg6GRE1Mo4IkTkRbKqzmG/6ST2mU5iX/lJHKvMgQ0O0bG8j8UM27FDsB07VO1mVXQsNB07Q9uxEzRpXaBul8iRIyIfxyJE5KHK7VV/FZ4s7DedxAHTKRTbTaJj+TR7zhnYc86gau1KAIDCYIQmNf2vctQZmo6doPTzF5ySiFyJRYjIQzgkBw6YsrG59DC2lB7GAVM2R3sEkyorYNmzA5Y9O2ACAKUS6sRk6Hr0gbZHX2jTukCh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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "PREPROCESSING (Data Cleaning)\n",
        "\n",
        "We check for missing values.\n",
        "\n",
        " Missing data can cause errors and reduce model accuracy.\n",
        "\n",
        " In our dataset: no missing values were found (already clean).\n"
      ],
      "metadata": {
        "id": "5ZqEFSmn9qbd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "print(\"Preprocessing — Checking Missing Values\")\n",
        "\n",
        "\n",
        "print(\"\\nMissing values per column:\")\n",
        "print(df.isnull().sum())\n",
        "\n",
        "missing_total = df.isnull().sum().sum()\n",
        "if missing_total == 0:\n",
        "    print(\"\\nNo missing values found — dataset is clean \")\n",
        "else:\n",
        "    print(f\"\\nFound {missing_total} missing values — dropping rows\")\n",
        "    df = df.dropna()\n",
        "    print(f\"Dataset size after cleaning : {df.shape[0]:,} rows\")\n",
        "\n",
        "print(f\"\\nFinal dataset size : {df.shape[0]:,} rows\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VzqyH3yq9ysZ",
        "outputId": "20487a9f-f0a1-4604-9028-cb22e4c77c27"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Preprocessing — Checking Missing Values\n",
            "\n",
            "Missing values per column:\n",
            "text     0\n",
            "label    0\n",
            "dtype: int64\n",
            "\n",
            "No missing values found — dataset is clean \n",
            "\n",
            "Final dataset size : 39,787 rows\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "FEATURE EXTRACTION (TF-IDF)\n",
        "\n",
        " Machines cannot read text directly — we must convert\n",
        " words into numbers using TF-IDF (Term Frequency–Inverse\n",
        " Document Frequency).\n",
        "\n",
        "\n",
        " TF  = how often a word appears in one ingredient list\n",
        " IDF = how rare/unique that word is across all ingredients\n",
        "\n",
        "\n",
        " max_features=5000  → keep only top 5000 important words\n",
        " ngram_range=(1, 2) → capture single words AND two-word\n",
        "                      phrases like \"pork fat\", \"beef stock\""
      ],
      "metadata": {
        "id": "tznkBdN8-Ibt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "print(\"Feature Extraction — TF-IDF Vectorization\")\n",
        "\n",
        "\n",
        "X = df['text']    # Input  : ingredient text\n",
        "y = df['label']   # Output : halal or haram\n",
        "\n",
        "print(f\"\\n  Total text samples : {X.shape[0]:,}\")\n",
        "\n",
        "tfidf = TfidfVectorizer(\n",
        "    stop_words='english',   # Remove common words like \"the\", \"and\"\n",
        "    max_features=5000,      # Keep top 5000 words only\n",
        "    ngram_range=(1, 2)      # Single words + two-word phrases\n",
        ")\n",
        "\n",
        "X_tfidf = tfidf.fit_transform(X)\n",
        "print(f\"  TF-IDF matrix shape : {X_tfidf.shape}\")\n",
        "print(f\"   {X_tfidf.shape[0]:,} samples  |  {X_tfidf.shape[1]:,} word features\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "piSKzk8K-lkk",
        "outputId": "46a2a88d-eea4-4678-ac44-544386c6089d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Feature Extraction — TF-IDF Vectorization\n",
            "\n",
            "  Total text samples : 39,787\n",
            "  TF-IDF matrix shape : (39787, 5000)\n",
            "   39,787 samples  |  5,000 word features\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "TRAIN / TEST SPLIT\n",
        "\n",
        "We split data into:\n",
        "\n",
        "  80% Training   → the model LEARNS from this\n",
        "\n",
        "  20% Testing    → we EVALUATE the model on unseen data\n",
        "\n",
        " random_state=42 → ensures the same split every run\n",
        "\n",
        "stratify=y      → keeps halal/haram ratio equal in both sets"
      ],
      "metadata": {
        "id": "LEZAWT3b-54t"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "print(\" Train / Test Split  (80% / 20%)\")\n",
        "\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X_tfidf, y,\n",
        "    test_size=0.2,\n",
        "    random_state=32,\n",
        "    stratify=y      # Keep label balance in both splits\n",
        ")\n",
        "\n",
        "print(f\"\\n  Training samples : {X_train.shape[0]:,}  (model learns from these)\")\n",
        "print(f\"  Testing  samples : {X_test.shape[0]:,}  (model is evaluated on these)\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FLO_eBhC_EzY",
        "outputId": "2caac0d0-6afd-4a52-be0e-91349d3d5631"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " Train / Test Split  (80% / 20%)\n",
            "\n",
            "  Training samples : 31,829  (model learns from these)\n",
            "  Testing  samples : 7,958  (model is evaluated on these)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "TRAIN MODEL 1 : Naive Bayes (MultinomialNB)\n",
        "\n",
        " Naive Bayes is a probability-based algorithm.\n",
        "\n",
        " It answers: \"Given these words, what is the probability\n",
        " this ingredient list is halal vs haram?\"\n",
        "\n",
        " Why Naive Bayes for text?\n",
        "\n",
        " Fast to train\n",
        "\n",
        " Works great with word counts / TF-IDF\n",
        "\n",
        " Performs well even on large datasets"
      ],
      "metadata": {
        "id": "ELeuBCs6_Ohj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "print(\"Training Model 1 — Naive Bayes\")\n",
        "\n",
        "\n",
        "nb_model  = MultinomialNB()\n",
        "nb_model.fit(X_train, y_train)         # Train on 80%\n",
        "y_pred_nb = nb_model.predict(X_test)  # Predict on 20%\n",
        "\n",
        "nb_accuracy = accuracy_score(y_test, y_pred_nb)\n",
        "\n",
        "print(f\"\\n  Naive Bayes trained successfully\")\n",
        "print(f\"\\n  Accuracy : {nb_accuracy * 100:.2f}%\")\n",
        "print(\"\\n  Classification Report:\")\n",
        "print(classification_report(y_test, y_pred_nb))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0f4CAiqs_dJF",
        "outputId": "7afd2f4d-ba2c-4e16-fbe1-4c026dc828fa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Model 1 — Naive Bayes\n",
            "\n",
            "  Naive Bayes trained successfully\n",
            "\n",
            "  Accuracy : 86.53%\n",
            "\n",
            "  Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "       halal       0.88      0.87      0.88      4366\n",
            "       haram       0.84      0.86      0.85      3592\n",
            "\n",
            "    accuracy                           0.87      7958\n",
            "   macro avg       0.86      0.86      0.86      7958\n",
            "weighted avg       0.87      0.87      0.87      7958\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "TRAIN MODEL 2 : Decision Tree\n",
        "\n",
        " A Decision Tree makes predictions by asking yes/no questions\n",
        "\n",
        " about specific words in the ingredient list.\n",
        "\n",
        " Example: \"Does it contain 'pork'?\" → Yes → likely Haram\n",
        "\n",
        "\n",
        " Why Decision Tree?\n",
        "\n",
        " Easy to understand and explain\n",
        "\n",
        " No scaling required\n",
        "\n",
        " Good baseline classifier"
      ],
      "metadata": {
        "id": "AVUahit2_zio"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "print(\"Training Model 2 — Decision Tree\")\n",
        "\n",
        "\n",
        "dt_model  = DecisionTreeClassifier(random_state=42)\n",
        "dt_model.fit(X_train, y_train)\n",
        "y_pred_dt = dt_model.predict(X_test)\n",
        "\n",
        "dt_accuracy = accuracy_score(y_test, y_pred_dt)\n",
        "\n",
        "print(f\"\\n  Decision Tree trained successfully\")\n",
        "print(f\"\\n  Accuracy : {dt_accuracy * 100:.2f}%\")\n",
        "print(\"\\n  Classification Report:\")\n",
        "print(classification_report(y_test, y_pred_dt))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "we3blvAdABKO",
        "outputId": "53067a80-d094-4c31-953c-695f24d07ecc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Model 2 — Decision Tree\n",
            "\n",
            "  Decision Tree trained successfully\n",
            "\n",
            "  Accuracy : 99.18%\n",
            "\n",
            "  Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "       halal       0.99      0.99      0.99      4366\n",
            "       haram       0.99      0.99      0.99      3592\n",
            "\n",
            "    accuracy                           0.99      7958\n",
            "   macro avg       0.99      0.99      0.99      7958\n",
            "weighted avg       0.99      0.99      0.99      7958\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Confusion matrix for the best model\n",
        "cm = confusion_matrix(y_test, best_preds, labels=['halal', 'haram'])\n",
        "\n",
        "plt.figure(figsize=(6, 5))\n",
        "sns.heatmap(\n",
        "    cm,\n",
        "    annot=True,\n",
        "    fmt='d',\n",
        "    cmap='Greens',\n",
        "    xticklabels=['halal', 'haram'],\n",
        "    yticklabels=['halal', 'haram'],\n",
        "    linewidths=0.5,\n",
        "    linecolor='gray'\n",
        ")\n",
        "plt.title(f'Confusion Matrix — {winner_name}',\n",
        "          fontsize=14, fontweight='bold')\n",
        "plt.ylabel('Actual Label', fontsize=12)\n",
        "plt.xlabel('Predicted Label', fontsize=12)\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 507
        },
        "id": "p8TISM_Pdvv5",
        "outputId": "d95186c9-8c56-4658-d048-b310d90d497a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x500 with 2 Axes>"
            ],
            "image/png": "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1qyp8rhVq1YhKCgI9vb2Sn+WVX5+Pj788EPxuXVdu3bFuHHjADydCLthwwZxGG779u1YsmSJUl9f2XR1dWFmZgYXFxd069YNYWFhSE9PR2hoaIXv3SeffIJTp05h1KhRaNiwIYyNjaGnpwd7e3t07twZX375pcIcIuDpVV5JSUmYP38+fHx8YGVlBT09PdSqVQtvvvnmc+eDAU/n4ixZsgQffPABPD09YW1tDV1dXZibm6N169aYPXs2YmNjoaf3/EK9rq4uoqOjsWnTJvTu3Rt2dnbQ09ODhYUF2rVrh3nz5uHEiRNixfJ1oqOjg59++gnbt29HQEAAateuLf674uzsjL59+2LhwoX45ZdfFI57++23cfbsWYSEhKBp06YwNTWFrq4ubGxs4O3tjSlTpuDw4cMqeRQRqQc+C4yIiIi0DitAREREpHWYABEREZHWYQJEREREWocJEBEREWkdJkBERESkdZgAERERkdZhAkRERERaRysehfG8B28SERG9TDNnznxlryV7q7ZS+xNiriu1P3WhFQkQAMw6tErVIRC9tmb5jMS00CmqDoOISGm0JgEiIiLSCjKZqiN4LTABIiIi0iSc3SsJ3yYiIiLSOqwAERERaRIOgUnCChARERFpHVaAiIiINAkLQJIwASIiItIkHAKThENgREREpHVYASIiItIkLG1IwgSIiIhIk3AITBLmiURERKR1WAEiIiLSJCwAScIEiIiISJPoMAOSgkNgREREpHVYASIiItIkLABJwgSIiIhIk/AqMEk4BEZERERahxUgIiIiTcICkCSsABEREZHWYQWIiIhIk/AyeEmYABEREWkS5j+ScAiMiIiItA4rQERERJqEl8FLwgSIiIhIk3AOkCQcAiMiIiKtwwSIiIhIk8iUvNTQN998A5lMhokTJ4rbnjx5guDgYNjY2MDU1BQBAQHIyspSOC4jIwP+/v4wNjaGnZ0dpkyZgqKiIoU2cXFxaNmyJeRyOdzc3LBu3bpqx8cEiIiISJPIZMpdauDEiRNYsWIFvLy8FLZPmjQJW7duxaZNmxAfH48bN25gwIAB4v7i4mL4+/ujoKAAR44cwfr167Fu3TrMmDFDbJOWlgZ/f3906dIFSUlJmDhxIkaMGIHdu3dXK0YmQERERKQ0eXl5GDx4MFatWgUrKytxe05ODtasWYP58+eja9euaNWq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SofjanpPnFcRY3WkpKQgNDQU+/btQ25ursK+nJwchXUnJ6dyE80bNmwIAEhPT0f79u3x77//4vz587C1ta3w9UrPj4g0HxMgompq27ateBVYZeRyebmkqKSkBHZ2doiKiqrwmMp+Kb9K6hTj/fv30blzZ5ibmyM8PBz169eHoaEhTp48iWnTpqGkpKTafZaUlKBp06aYP39+hfvr1KnzomET0WuCCRDRK1K/fn3s3bsXb775psLQzrOcnZ0BPK3GuLq6ittv3bpV7kqsil4DAM6ePYvu3btX2q6y4bBXEaNUcXFxuHPnDv7880906tRJ3F56td2zbty4Ue52A5cuXQLw9K7OwNPzO336NLp16yZpSJCINBfnABG9Iu+//z6Ki4sxe/bscvuKiopw//59AE/nGOnr62Px4sUQBEFss3Dhwue+RsuWLeHi4oKFCxeK/ZUq21dpkvBsm1cRo1S6urrl4i4oKMDSpUsrbF9UVIQVK1YotF2xYgVsbW3RqlUrAE/P77///sOqVavKHf/48WM8fPhQafETkXpjBYjoFencuTNGjx6NiIgIJCUloUePHtDX18e///6LTZs2YdGiRXj33Xdha2uLyZMnIyIiAn369EHv3r1x6tQp7Ny5E7Vq1aryNXR0dLBs2TL07dsXzZs3x9ChQ+Ho6IgLFy4gJSUFu3fvBgAxIfj000/h5+cHXV1dDBo06JXEWNY///yDr776qtx2X19fdOjQAVZWVggKCsKnn34KmUyGDRs2KCREZTk5OeHbb79Feno6GjZsiN9++w1JSUlYuXKleOn+kCFDEB0djTFjxmD//v148803UVxcjAsXLiA6Ohq7d+9+7vAmEWkIlV6DRvQaKb0M/sSJE1W2CwoKEkxMTCrdv3LlSqFVq1aCkZGRYGZmJjRt2lSYOnWqcOPGDbFNcXGxEBYWJjg6OgpGRkaCr6+vcPbs2XKXZj97GXypQ4cOCW+99ZZgZmYmmJiYCF5eXsLixYvF/UVFRcL48eMFW1tbQSaTlbskXpkxVgZApcvs2bMFQRCEw4cPC+3btxeMjIwEJycnYerUqcLu3bvLnXPnzp2Fxo0bC//884/g7e0tGBoaCs7OzsKSJUvKvW5BQYHw7bffCo0bNxbkcrlgZWUltGrVSggLCxNycnLEdrwMnkizyQShkj+niIiIiDQU5wARERGR1mECRERERFqHCRARERFpHSZAREREpHWYABEREZHWYQJEREREWocJEBEREWkdJkBERESkdZgAERERkdZhAkRERERahwkQERERaR0mQERERKR1mAARERGR1vl/H6pRvcdqO60AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "print(\" Model Comparison\")\n",
        "\n",
        "\n",
        "print(f\"\\n  Naive Bayes   : {nb_accuracy * 100:.2f}%\")\n",
        "print(f\"  Decision Tree : {dt_accuracy * 100:.2f}%\")\n",
        "\n",
        "if nb_accuracy >= dt_accuracy:\n",
        "    winner_name  = \"Naive Bayes\"\n",
        "    winner_acc   = nb_accuracy\n",
        "    best_preds   = y_pred_nb\n",
        "else:\n",
        "    winner_name  = \"Decision Tree\"\n",
        "    winner_acc   = dt_accuracy\n",
        "    best_preds   = y_pred_dt\n",
        "\n",
        "print(f\"\\n   WINNER : {winner_name}  ({winner_acc * 100:.2f}%)\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7XzlQ2RLAK7H",
        "outputId": "5c6a1b7a-bd05-4584-bfe4-35aa102207f4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " Model Comparison\n",
            "\n",
            "  Naive Bayes   : 86.53%\n",
            "  Decision Tree : 99.18%\n",
            "\n",
            "   WINNER : Decision Tree  (99.18%)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "VISUALIZATION 2 :\n",
        "\n",
        " Accuracy Bar Chart\n",
        "Visual comparison of both models' accuracy scores.\n",
        "\n",
        " Makes it easy to see which model performed better."
      ],
      "metadata": {
        "id": "fQV2qrTXAot0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "models     = ['Naive Bayes', 'Decision Tree']\n",
        "accuracies = [nb_accuracy * 100, dt_accuracy * 100]\n",
        "colors_bar = ['#2ecc71', '#3498db']\n",
        "\n",
        "plt.figure(figsize=(8, 6))\n",
        "bars = plt.bar(models, accuracies, color=colors_bar,\n",
        "               width=0.5, edgecolor='black')\n",
        "\n",
        "for bar, acc in zip(bars, accuracies):\n",
        "    plt.text(\n",
        "        bar.get_x() + bar.get_width() / 2,\n",
        "        bar.get_height() + 0.3,\n",
        "        f'{acc:.2f}%',\n",
        "        ha='center', va='bottom',\n",
        "        fontsize=13, fontweight='bold'\n",
        "    )\n",
        "\n",
        "plt.title('Model Accuracy Comparison\\nHalal vs Haram Classification',\n",
        "          fontsize=14, fontweight='bold')\n",
        "plt.ylabel('Accuracy (%)', fontsize=12)\n",
        "plt.xlabel('Machine Learning Model', fontsize=12)\n",
        "plt.ylim(0, 110)\n",
        "plt.grid(axis='y', linestyle='--', alpha=0.6)\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 607
        },
        "id": "nG5OHhX-AvDl",
        "outputId": "af0c1539-44bf-49dd-ebb7-e91d14f83bc8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x600 with 1 Axes>"
            ],
            "image/png": "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YRiaNGmSmjRpokceecTu12bbtm3av3+/+fzhhx9WUFCQJk+eLEk6f/68fvzxR91///028/Xr10/bt283n9etW1edO3eWq6urdu7cqa1bt5rTjh8/rl69eikxMVGSzMH0jRo10oULF/TLL7/Y3Y/sDh06JEnq0aOHGjZsqOPHj8vHx0eSVK5cOd1///2qXr26/P395ejoqNOnT2vhwoW6ePGiTp8+rTfffNP8dTo9PV3dunUzlyllnJffvn17paena+vWrYqLi5Mkubm56dFHH9W7774rSfrss880fPhwc77Fixebjx955BG5urrm24/t27eb71NJ6t27t8qUKVOg1+DLL7+0OfWvXr166t69u86cOaO5c+cqPT1dp0+fVo8ePbRv3z45OeX82t25c6fq1aunHj16aOXKleb/fO7cuZKk22+/Xffdd5+++uor8/WZOnWqxo4dKxcXlxzL++uvv+Tj46NnnnlGFotFs2bN0uXLlyVJL774oh544AHVqFFDklSmTBmFh4crLCxM/v7+cnd318WLF/XDDz/owIEDSklJ0VNPPWXz/s0qv/dAXmbMmGE+joiIUNu2bZWQkKCTJ09q48aNunr1qjk9LS1N3bt316lTpyRlHPXs16+fKlWqpGXLlmnv3r3m69G4cWP1798/13Vu3LhR3bt312233aZ58+aZR+HmzZund9991+YIIgA7FHdlA6B0yv6r9vLly42XX37ZfL5y5UqjRo0ahiSjbNmyRlJSUp5HLCZPnmyzrBUrVpjTzp8/b3h6eprTunbtahiGYaSmptrE27RpY6SnpxuGYRhWq9Xo0KFDrr+6X7x40XBycjLjs2bNsunX8OHDbX5Jzqu/13PE4oknnrD5BT05OdkwDMOoXr26Ge/Ro4fNPHv27LFZX+fOnY2UlBSbNocPHzYfZ//VfN68eTZt09PT8zzyUNgjFpKMKVOm5NnvhIQEY/Xq1cann35qTJo0yXj//feNrl27mvOGhoaabb/77jub5Q4bNsywWq159vf48eOGo6Oj2X7Hjh2GYRjGpUuXDGdn5xzx/CxatMhm3TNmzLjmPJkaNmxozle1alUjMTHRnDZ9+vRcf8k3DNsjFgEBAUZsbKxhGIZx8OBBm3nKlStnHj1ZuXKlzbQ9e/aYy8v+f/ntt9/Mab/99pvNtFdeecWmD+np6cbWrVuNOXPm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7b///nt16tRJgYGBcnFxUYUKFfTII49o796917XelJQUTZgwQV27dlWFChVksVjMv7Vr1+Y6T2pqqj766CO1bt1a/v7+cnJykqenp+rXr6/nnntO58+ft2l/+PBhPfjggwoODpaPj49at26tX375JcdyT548KS8vLzVu3FhpaWnX1Q+UfBbDMIziTqKkiYuLk4+Pj2JjY+Xt7V3c6QAAgFLuyJEjuuuuu3TmzJlcp7dv314rVqyQi4uLJOmll17Su+++m2tbV1dXffPNN+rcuXOB1n358mX5+fnlOu3XX39V27Ztc8R79uypb775Js9lhoSEaOfOnQoICFBCQoIaNGigI0eOaOLEiapXr5569eql9PR0bdu2TQ0bNjTnu//++/Xjjz9q69atatKkSYHyR/G6nv1ijlgAAADcYE888YRZVDRq1EjLli3TokWLVKNGDUnSmjVr9Pbbb0uSNm/ebFNUjB8/Xj///LOeffZZSVJycrL69eunS5cuFWjdDg4Oatq0qR5//HF9/vnn12z/999/2xQVjz/+uFatWqUPPvhAjo6OkjKOPCxatEiStGnTJh05ckQ+Pj56/vnn1aVLF0VERCglJUWLFy82l7No0SJ9//33GjVqFEXFLcqpuBMAAAC4lSUkJGj16tXm80mTJunuu++WlHHKUd++fSVJ06ZN09ixY/Xtt9+abdu0aaNXX31VknTPPfdo+fLlOnTokC5duqS5c+fqmWeeueb6vb29tX37dvP5kCFD8m1/+fJlm+fvv/++PD09FRERoVmzZmnfvn2SMk6xkjIKHUlydnaWxWKRlHFUJeu0y5cv6+mnn1a1atU0fvz4a+aM0okjFgAAADdQXFycrFar+bxMmTLmY09PT/PxxYsXtWfPHsXExOTaNnv7X3/99Uakq/r166t8+fLm8+eff15r1qzRpEmT9Oeff5p5dOvWTZLUsmVL+fn5KTo6WitWrNCJEyfM3Dp16mQu49y5c5oxY4Y8PDxuSN4ofhQWAAAAN1C5cuUUEBBgPv/ggw908eJFnTt3Th999JFN22PHjqlu3brm819//VW//vqrrl69qqVLl2r37t02bW8ENzc3rVixQo0bN5YkffLJJ4qIiNDo0aOVnp6uiIgIbd68WVWqVJEkBQQEaOnSpapdu7a6dOmiKlWqyGq1avr06WrXrp3Wr1+vzz//XH37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BAYG2tz/Q7p524i8puemRBUW8fHx+vvvv83nR48e1a5du8xDOKNGjdKbb76pmjVrqlq1anr11VdVoUIF86hG3bp1de+99+rRRx/VJ598otTUVI0cOVIPP/xwjhvOAACAonHixAnVrlNXSVcTr90YwHVzc/fQwT8P5CguSpoSVVj8/vvvuvvuu83nzz77rCRpwIABmjNnjl544QUlJCRo2LBhunz5slq3bq2VK1fKzc3NnGfevHkaOXKk2rdvLweHjBvkffjhhze9LwAA/FtER0cr6Wqiqg36j9zK1yjudIBbStLZv3V09nOKjo6msLgebdu2zXGuWFYWi0Xjx4/X+PHj82zj7++v+fPn34j0AABAPtzK11CZyvWKOw0AxST3k60AAAAA4DpQWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFSqXjx49r5MiRqlOnjsqUKSMnJyeVLVtWd999t2bNmiXDMHLMc/jwYT3xxBOqWbOm3N3d5evrq7p162rgwIE6fPhwgdZ77NgxWSyWfP++//57m3lmzpypLl26qFq1avLy8pKzs7PKli2rNm3a6MMPP1RqamqOPB988EEFBwfLx8dHrVu31i+//JIjl5MnT8rLy0uNGzdWWlradbx6AAAARc+puBMArtfx48fVuHFjXbp0ySYeHR2ttWvXau3atfr99981ffp0c9o333yjvn37KikpyYwlJSUpNjZWf/75px588EFVr179huQ7b948rVu3LkeuGzZs0IYNG7R69Wp99913kqSEhAR16NBBR44c0cSJE1WvXj316tVLnTp10rZt29SwYUNzGcOHD9fVq1c1c+ZMOTnxUQYAAMWLvRGUOjNnzjSLCm9vb02ZMkUVK1bUtGnTtHz5cknSp59+qvfee0+e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jvjvv/+ul156SVWrVpWLi4saNWqkoUOHaseOHUWaKAAAAICSq8CFxaJFi1S3bl19++23NvEmTZpo4sSJOnnypNLS0rR37159/vnnaty4cZEnCwAAAKBkKnBhsWPHDj3//PN69NFHFRERoX379kmSPvnkE50+fVpVqlSRq6urGjRoIEdHR82aNeuGJQ0AAACgZClwYWGxWPTYY4/p0KFDql+/vpo2baqRI0fK3d1dGzZs0PHjx7V582YdPXpU27ZtU7Vq1W5k3gAAAABKkOsevO3j46MpU6Zox44dOnTokGrUqKGPPvpIFStWVPPmzVWlSpUbkScAAACAEqzQV4W67bbb9NNPP2n27Nn66KOPFBYWplWrVhVlbgAAAABKiQIXFvHx8XriiSdUsWJF+fn56d5779X+/fv1wAMPaN++ferfv7969uypBx54QIcPH76ROQMAAAAoYQpcWAwfPlzfffed3n77bc2dO1dXr15V586dlZKSImdnZ40ZM0YHDx6Un5+fwsLC9MILL9zIvAEAAACUIAUuLH744Qe99NJLGjBggB544AF99tlnOnHihHl1KEkqX7685s6dq7Vr12rDhg03JGEAAAAAJU+BCwsfHx8dPXrUfH7s2DFZLBb5+PjkaNu8eXNt3ry5aDIEAAAAUOI5FbThmDFjNHz4cO3evVt+fn768ccf1aNHD4WGht7I/AAAAACUAgUuLB577DHVq1dPP/zwg65evar//ve/6tOnz43MDQAAAEApUeDCQpJat26t1q1b36hcAAAAAJRSBRpjkZiYWOgV2DMvAAAAgNKhQIVFSEiIxo8fr7NnzxZ4wadPn9Zrr72mypUrFzo5AAAAAKVDgU6FmjFjhl5//XWNHz9erVq1UkREhBo3bqxq1arJz89PhmEoJiZGR48e1e+//67Vq1dry5YtqlmzpqZPn36j+wAAAACgmBWosHjooYf04IMP6rvvvtOcOXP01ltvKSUlRRaLxaadYRhycXFRhw4dtGTJEj3wwANycCjwFW0BAAAAlFIFHrzt4OCgbt26qVu3bkpOTtaOHTv0559/6uLFi5KkgIAA1alTR02aNJGrq+sNSxgAAABAyXNdV4XK5OrqqjvvvFN33nlnUedzTadPn9aYMWP0448/KjExUTVq1NDs2bPVtGlTSRlHTcaNG6eZM2fq8uXLatWqlWbMmKGa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VqdOnSrqnAAAAACUMoUqLBo2bKhTp07pnXfeUZUqVeTq6ipvb2+bPx8fn6LOFQAAAEAJVahR1j179pTFYinqXAAAAACUUoUqLObMmVPEaQAAAAAozQp1KhQAAAAAZFWoIxZffPFFgdr179+/MIsHAAAAUMoUqrAYOHBgntOyjr2gsAAAAAD+HQpVWBw9ejRHLD09XceOHdP06dN14sQJzZ071+7kAAAAAJQOhSosqlSpkms8NDRU7dq1U5cuXTRt2jR9/PHHdiUHAAAAoHS4IYO377vvPi1cuPBGLBoAAABACXRDCovDhw8rOTn5RiwaAAAAQAlUqFOh1q9fn2v88uXLWr9+vT788EN169bNnrwAAAAAlCKFKizatm2b6523DcOQo6OjevXqpY8++sju5AAAAACUDoUqLH799dccMYvFIj8/P1WpUkXe3t52JwYAAACg9ChUYREeHl7UeQAAAAAoxQo1ePvo0aNavnx5ntOXL1+uY8eOFTYnAAAAAKVMoY5YPPfcc4qLi9P999+f6/SPP/5Yvr6++uqrr+xKDgAAAEDpUKgjFps3b9Y999yT5/T27dtrw4YNhU4KAAAAQOlSqMIiJiZGXl5eeU739PTUxYsXC50UAAAAgNKlUIVF5cqV9dtvv+U5fcOGDapUqVKhkwIAAABQuhSqsOjTp48WLFigDz/8UFar1Yynp6dr6tSpWrhwoR555JEiSxIAAABAyVaowdsvvfSSNm7cqFGjRumtt95S7dq1JUkHDx7UhQsX1LZtW73yyitFmigAAACAkqtQRyxcXV31888/6/PPP1fz5s0VHR2t6OhoNW/eXLNmzdLq1avl6upa1LkCAAAAKKEKdcRCkhwcHDRo0CANGjSoKPMBAAAAUAoV6ojFpUuXtGfPnjynR0ZGKiYmptBJAQAAAChdClVYPPPMMxo2bFie0x977DE999xzhU4KAAAAQOlSqMLil19+0QMPPJDn9Pvvv1+rV68udFIAAAAASpdCFRYXLlxQYGBgntMDAgIUFRVV6KQAAAAAlC6FKizKly+vnTt35jl9x44dKlu2bKGTAgAAAFC6FKqw6Natmz7//HN99913OaZ9++23mj17trp37253cgAAAABKh0Jdbvb111/X6tWr1b17dzVs2FD169eXJO3du1e7d+9W3bp19cYbbxRpogAAAABKrkIdsfDx8dGWLVs0duxYpaamasmSJVqyZIlSU1P16quvauvWrfL19S3iVAEAAACUVIW+QV6ZMmX0xhtv5HlkIiYmRn5+foVODAAAAEDpUagjFnlJTk7W4sWL1a1bN5UvX74oFw0AAACgBCv0EYtMhmFozZo1mjdvnpYuXaq4uDiVLVtWjzzySFHkBwAAAKAUKHRhsWPHDs2bN09fffWVzp07J4vFoocfflgjR47UHXfcIYvFUpR5AgAAACjBrquwOHLkiObNm6d58+bp0KFDqlixovr27avmzZurd+/e6tmzp1q2bHmjcgUAAABQQhW4sGjZsqW2bdumwMBAPfjgg/rss8/UunVrSdLhw4dvWIIAAAAASr4CFxZbt25VtWrVNGnSJHXp0kVOTnYPzwAAAABwiyjwVaGmTZum8uXLq3v37goODtZjjz2mX3/9VYZh3Mj8AAAAAJQCBS4shg8fro0bN+rw4cMaNWqUNmzYoPbt26tixYp67bXXZLFYGLANAAAA/Etd930sqlWrprFjx2r//v3avn27Hn74Ya1du1aGYWj48OEaNmyYvv/+eyUlJd2IfAEAAACUQHbdIK9JkyaaNGmSTp48qZ9//lkdO3bUwoUL9cADDygwMLCocgQAAABQwhXJnbcdHBwUERGhOXPm6Pz581qwYIHat29fFIsGAAAAUAoUSWGRlZubm3r37q1vv/22qBcNAAAAoIQq8sICAAAAwL8PhQUAAAAAu1FYAAAAALAbhQUAAAAAu5XqwuLdd9+VxWLRqFGjzFhSUpJGjBihgIAAeXp6qmfPnjp//nzxJQkAAAD8C5TawmL79u3673//qwYNGtjEn3nmGS1fvlyLFy/WunXrdObMGfXo0aOYsgQAAAD+HUplYREfH6++fftq5syZ8vPzM+OxsbH6/PPPNWnSJLVr105NmjTR7NmztWnTJm3ZsqUYMwYAAABubU7FnUBhjBgxQl26dFFERITefPNNM75jxw6lpqYqIiLCjNWpU0eVK1fW5s2bdccdd+S6vOTkZCUnJ5vP4+LiJEnp6elKT0+XJFksFjk4OMhqtcowDLNtZjyz3bXiDg4OslgsucYlmct3dnaWkxwkQ7JIcpTFpn2axZDFsI0bktIthhwMyaEAcaskq8WQg2GxqTCtMmS1SI6GxWatecXTZciwSE6GbY7pMmRIcsqeuwz6RJ+KpU9SxmfTMAybz6Cjo6MMw5DVarVp6+jomOMzn1f8Zm4jChKnT/TpZvbJ/N4yP3KGHGWbe7ocZJEhhwLEDVlklUUOMmTJErfKIkMWOco2l7zi6bJIecaVS455xR3oE30qtj5ZJDk7O9t8d93MbUT26fkpdYXFV199pT/++EPbt2/PMe3cuXNycXGRr6+vTTwoKEjnzp3Lc5nvvPOO3njjjRzxffv2ydPTU5Lk7++vypUr69SpU7p06ZLZJjg4WMHBwTp27JiuXLlixkNCQhQQEKBDhw4pKSnJjIeGhsrb21v79++3+UfVrl1bLi4uioyMVExMjIYOHSpv1xpapdNytzqrbVwVs22axaoffQ8rMM1Dd8RXNOPxDin61ee4KqV4q2FikBm/4JSoLV6nVTMpQLWS/M34CZc47S5zXmGJ5VQ5xduM/+V2SQfdL6pZfAWVTfMw47s9zuuEa5zaxFWWp9XFjG/xPK0Lzom6JzZUTsY/u4lrvY/rqkOqOl2uYfO6/uj7N32iT8XSp68VJV9fX8XExCgyMlJSxkY4LCxMV65c0ZEjR8y2bm5uqlOnjmJiYnTy5Ekz7uXlperVqysqKspmu3IztxFZhYWFKSUlRQcPHjRj9Ik+3ew+ZX5veYZIv0oKdEpWc49/com3Oml9fDlVck5UmHusGb+Q5qrtiQGq4Rqvmq7/5HgyxUORSb6q5xarEJdEM34o2UuHkr3U2CNGZZ3++UEw8qqPTqaWUSvPaHk6pJnxbYn+ik5zU3uv83Ky/LNDtT6+rK5aHdXR23bf4Ke4YLk7pKuN5wUzlmZY9POV8vSJPhVbn8q6S+FDh5rfXTd7GxEfH6+CshjZS50S7OTJk2ratKlWrVpljq1o27atGjVqpClTpmj+/PkaNGiQzdEHSWrevLnuvvtuTZw4Mdfl5nbEIiQkRJcuXZK3d8ZOz838RWjXrl1q2bKlqs/uLZfaZW+5X41vxV/C6VPp6FPcwXM63G+Btm/frkaNGpnxW+VX4+w50if6dLP6lPm9VeeFxXINqadb8Vdj+kSfiqtPiSf26dD7vbR582bzu+tmbiPi4uLk7++v2NhYc784L6XqiMWOHTsUFRWlxo0bm7H09HStX79e06ZN008//aSUlBRdvnzZ5qjF+fPnFRwcnOdyXV1d5erqmiPu6OgoR0dHm1jmhjS3tkUVt1gsSk1NVZqscrFk7JylZXujSZJhyT1utWTsiBU8bmR722dIt+Rsm188La94brnnFadP9Ek3uE+GIYvFkuMzmFtMyvszf73xotxGFDROn+hTUeV4rbj5vWVIGd+mFnNHKSvjOuPW/79zll16HkNErz+ec9l5x+kTfSqePhmSUlNTc2wrbtY2Iq/puSlVhUX79u1zHIodNGiQ6tSpozFjxigkJETOzs5as2aNevbsKUk6ePCgTpw4oZYtWxZHygAAAMC/QqkqLLy8vFS/fn2bWJkyZRQQEGDGhwwZomeffVb+/v7y9vbWk08+qZYtW+Y5cBsAAACA/UpVYVEQkydPloODg3r27Knk5GR17NhR06dPL+60AAAAgFtaqS8s1q5da/Pczc1NH3/8sT7++OPiSQgAAAD4FyqVN8gDAAAAULJQWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALtRWAAAAACwG4UFAAAAALuVqsLinXfeUbNmzeTl5aVy5cqpW7duOnjwoE2bpKQkjRgxQgEBAfL09FTPnj11/vz5YsoYAAAA+HcoVYXFunXrNGLECG3ZskWrVq1SamqqOnTooISEBLPNM888o+XLl2vx4sVat26dzpw5ox49ehRj1gAAAMCtz6m4E7geK1eutHk+Z84clStXTjt27FCbNm0UGxurzz//XPPnz1e7du0kSbNnz1bdunW1ZcsW3XHHHcWRNgAAAHDLK1VHLLKLjY2VJPn7+0uSduzYodTUVEVERJht6tSpo8qVK2vz5s3FkiMAAADwb1CqjlhkZbVaNWrUKLVq1Ur169eXJJ07d04uLi7y9fW1aRsUFKRz587luazk5GQlJyebz+Pi4iRJ6enpSk9PlyRZLBY5ODjIarXKMAyzbWY8s9214g4ODrJYLLnGM/tlGIacnZ3lJAfJkCySHGWxaZ9mMWQxbOOGpHSLIQdDcihA3CrJajHkYFhsKkyrDFktkqNhsVlrXvF0GTIskpNhm2O6DBmSnLLnLoM+0adi6ZOU8dk0DMPmM+jo6CjDMGS1Wm3aOjo65vjM5xW/mduIgsTpE326mX0yv7fMj5whR9nmni4HWWTIoQBxQxZZZZGDDFmyxK2yyJBFjrLNJa94uixSnnHlkmNecQf6RJ+KrU8WSc7OzjbfXTdzG5F9en5KbWExYsQI7d27Vxs3brR7We+8847eeOONHPF9+/bJ09NTUsZRkcqVK+vUqVO6dOmS2SY4OFjBwcE6duyYrly5YsZDQkIUEBCgQ4cOKSkpyYyHhobK29tb+/fvt/lH1a5dWy4uLoqMjFRMTIyGDh0qb9caWqXTcrc6q21cFbNtmsWqH30PKzDNQ3fEVzTj8Q4p+tXnuCqleKthYpAZv+CUqC1ep1UzKUC1kvzN+AmXOO0uc15hieVUOcXbjP/ldkkH3S+qWXwFlU3zMOO7Pc7rhGuc2sRVlqfVxYxv8TytC86Juic2VE7GP7uJa72P66pDqjpdrmHzuv7o+zd9ok/F0qevFSVfX1/FxMQoMjJSUsZGOCwsTFeuXNGRI0fMtm5ubqpTp45iYmJ08uRJM+7l5aXq1asrKirK5geLm7mNyCosLEwpKSk2F7KgT/TpZvcp83vLM0T6VVKgU7Kae/yTS7zVSevjy6mSc6LC3GPN+IU0V21PDFAN13jVdP0nx5MpHopM8lU9t1iFuCSa8UPJXjqU7KXGHjEq6/TPD4KRV310MrWMWnlGy9MhzYxvS/RXdJqb2nudl5Plnx2q9fFlddXqqI7etj86/hQXLHeHdLXxvGDG0gyLfr5Snj7Rp2LrU1l3KXzoUPO762ZvI+Lj41VQFiN7qVMKjBw5Ut9++63Wr1+vatWqmfFffvlF7du3V0xMjM1RiypVqmjUqFF65plncl1ebkcsQkJCdOnSJXl7Z+z03MxfhHbt2qWWLVuq+uzecqld9pb71fhW/CWcPpWOPsUdPKfD/RZo+/btatSokRm/VX41zp4jfaJPN6tPmd9bdV5YLNeQeroVfzWmT/SpuPqUeGKfDr3fS5s3bza/u27mNiIuLk7+/v6KjY0194vzUqqOWBiGoSeffFJLly7V2rVrbYoKSWrSpImcnZ21Zs0a9ezZU5J08OBBnThxQi1btsxzua6urnJ1dc0Rd3R0lKOjo00sc0OaW9uiilssFqWmpipNVrlYMnbO0rK90STJsOQet1oydsQKHjeyve0zpFtyts0vnpZXPLfc84rTJ/qkG9wnw5DFYsnxGcwtJuX9mb/eeFFuIwoap0/0qahyvFbc/N4ypIxvU4u5o5SVcZ1x6//fOcsuPY8hotcfz7nsvOP0iT4VT58MSampqTm2FTdrG5HX9NyUqsJixIgRmj9/vr799lt5eXmZh3l8fHzk7u4uHx8fDRkyRM8++6z8/f3l7e2tJ598Ui1btuSKUAAAAMANVKoKixkzZkiS2rZtaxOfPXu2Bg4cKEmaPHmyHBwc1LNnTyUnJ6tjx46aPn36Tc4UAAAA+HcpVYVFQYaDuLm56eOPP9bHH398EzICAAAAIJXy+1gAAAAAKBkoLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADYjcICAAAAgN0oLAAAAADY7ZYtLD7++GNVrVpVbm5uatGihbZt21bcKQEAAAC3rFuysFi4cKGeffZZjRs3Tn/88YcaNmyojh07KioqqrhTAwAAAG5Jt2RhMWnSJD366KMaNGiQbrvtNn3yySfy8PDQrFmzijs1AAAA4JZ0yxUWKSkp2rFjhyIiIsyYg4ODIiIitHnz5mLMDAAAALh1ORV3AkUtOjpa6enpCgoKsokHBQXpzz//zHWe5ORkJScnm89jY2MlSTExMUpPT5ckWSwWOTg4yGq1yjAMs21mPLPdteIODg6yWCy5xiXJarXqypUrcnJyUuqf0YpPTJUkOcpi0z5dRo64IckqQxZZbCrGvOJWSUY+cQdZbNaaVzy3XIoqTp/oU1H26erxS5KkK1euKCYmxow7OjrKMAxZrVab9o6Ojjk+83nFb+Y2oiBx+kSfbmafMr+3kk/uU3pyoiySHLN8/AxJ6UbGr5kOBYhbJVmNjJjNNsLImOZoke02Io94upGxDifbTUGe8TRDOXLPK06f6NPN6lPKuSNycnKy+e66mduIuLi4jNcs27pyc8sVFoXxzjvv6I033sgRr1q16s1PJovjb/5crOsHblVt27Yt7hSAW9KRL18p7hSAW1Zxf3dduXJFPj4++ba55QqLwMBAOTo66vz58zbx8+fPKzg4ONd5XnrpJT377LPmc6vVqkuXLikgIEAWiyXXeYBMcXFxCgkJ0cmTJ+Xt7V3c6QAAkC++t3A9DMPQlStXVKFChWu2veUKCxcXFzVp0kRr1qxRt27dJGUUCmvWrNHIkSNzncfV1VWurq42MV9f3xucKW413t7ebKABAKUG31soqGsdqch0yxUWkvTss89qwIABatq0qZo3b64pU6YoISFBgwYNKu7UAAAAgFvSLVlY9O7dWxcuXNBrr72mc+fOqVGjRlq5cmWOAd0AAAAAisYtWVhI0siRI/M89QkoSq6urho3blyO0+kAACiJ+N7CjWIxCnLtKAAAAADIxy13gzwAAAAANx+FBQAAAAC7UVjgX6Nt27YaNWpUcacBAEChVa1aVVOmTCnytkBRoLBAiTZw4EBZLBa9++67NvFly5Zd980Lv/nmG02YMKEo08shM9/Mv4CAAN17773as2fPDV0vAKD4ZN32Ozs7KygoSPfcc49mzZolq9VapOvavn27hg0bVuRtCyP7d172v6pVq96wdaNkorBAiefm5qaJEycqJibGruX4+/vLy8uriLLK27333quzZ8/q7NmzWrNmjZycnHTffffd8PUCAIpP5rb/2LFj+vHHH3X33Xfr6aef1n333ae0tLQiW0/ZsmXl4eFR5G0LY+rUqeb33dmzZyVJs2fPNp9v377dpn1KSsoNywUlA4UFSryIiAgFBwfrnXfeybPNxYsX1adPH1WsWFEeHh4KCwvTggULbNpkPRXq5ZdfVosWLXIsp2HDhho/frz5/LPPPlPdunXl5uamOnXqaPr06dfM19XVVcHBwQoODlajRo304osv6uTJk7pw4YLZZsyYMapVq5Y8PDwUGhqqV199VampqZKkY8eOycHBQb///rvNcqdMmaIqVaqYv37t3btXnTp1kqenp4KCgtSvXz9FR0eb7ZcsWaKwsDC5u7srICBAERERSkhIuGb+AIDrl7ntr1ixoho3bqyXX35Z3377rX788UfNmTPHbHf58mUNHTpUZcuWlbe3t9q1a6fdu3fbLGv58uVq1qyZ3NzcFBgYqO7du5vTsp7eZBiGXn/9dVWuXFmurq6qUKGCnnrqqVzbStKJEyfUtWtXeXp6ytvbWw899JDOnz9vTn/99dfVqFEjffnll6patap8fHz08MMP68qVK7n22cfHx/y+Cw4OliT5+vqaz5s1a6YJEyaof//+8vb2No+ebNy4UXfddZfc3d0VEhKip556yub7KTk5Wc8995wqVqyoMmXKqEWLFlq7du11/T9QPCgsUOI5Ojrq7bff1kcffaRTp07l2iYpKUlNmjTRDz/8oL1792rYsGHq16+ftm3blmv7vn37atu2bTp8+LAZ27dvn/bs2aNHHnlEkjRv3jy99tpreuutt3TgwAG9/fbbevXVVzV37twC5x4fH6///e9/qlGjhgICAsy4l5eX5syZo/3792vq1KmaOXOmJk+eLCnjiyAiIkKzZ8+2Wdbs2bM1cOBAOTg46PLly2rXrp1uv/12/f7771q5cqXOnz+vhx56SJJ09uxZ9enTR4MHD9aBAwe0du1a9ejRQ1xdGgBunnbt2qlhw4b65ptvzFivXr0UFRWlH3/8UTt27FDjxo3Vvn17Xbp0SZL0ww8/qHv37urcubN27typNWvWqHnz5rku/+uvv9bkyZP13//+V4cOHdKyZcsUFhaWa1ur1aquXbvq0qVLWrdunVatWqUjR46od+/eNu0OHz6sZcuW6fvvv9f333+vdevW5Tgd+Xr85z//UcOGDbVz5069+uqrOnz4sO6991717NlTe/bs0cKFC7Vx40abe4+NHDlSmzdv1ldffaU9e/aoV69euvfee3Xo0KFC54GbxABKsAEDBhhdu3Y1DMMw7rjjDmPw4MGGYRjG0qVLjWu9fbt06WKMHj3afB4eHm48/fTT5vOGDRsa48ePN5+/9NJLRosWLczn1atXN+bPn2+zzAkTJhgtW7bMN19HR0ejTJkyRpkyZQxJRvny5Y0dO3bkm+v7779vNGnSxHy+cOFCw8/Pz0hKSjIMwzB27NhhWCwW4+jRo2YeHTp0sFnGyZMnDUnGwYMHjR07dhiSjGPHjuW7XgCA/bJ+V2XXu3dvo27duoZhGMaGDRsMb29vc9ueqXr16sZ///tfwzAMo2XLlkbfvn3zXFeVKlWMyZMnG4ZhGB988IFRq1YtIyUl5Zptf/75Z8PR0dE4ceKEOX3fvn2GJGPbtm2GYRjGuHHjDA8PDyMuLs5s8/zzz9t8N+ZHkrF06VKb9Xfr1s2mzZAhQ4xhw4bZxDZs2GA4ODgYV69eNY4fP244Ojoap0+ftmnTvn1746WXXipQHig+HLFAqTFx4kTNnTtXBw4cyDEtPT1dEyZMUFhYmPz9/eXp6amffvpJJ06cyHN5ffv21fz58yVlHE5esGCB+vbtK0lKSEjQ4cOHNWTIEHl6epp/b775ps1Rjtzcfffd2rVrl3bt2qVt27apY8eO6tSpk44fP262WbhwoVq1aqXg4GB5enpq7NixNrl269ZNjo6OWrp0qSRpzpw5uvvuu82BcLt379avv/5qk1udOnUkZfza1LBhQ7Vv315hYWHq1auXZs6cafcYFQDA9TMMw7zYyO7duxUfH6+AgACb7ffRo0fN75Zdu3apffv2BVp2r169dPXqVYWGhurRRx/V0qVL8xzPceDAAYWEhCgkJMSM3XbbbfL19bX5Xq1atarNeMTy5csrKirquvudqWnTpjbPd+/erTlz5tj0v2PHjrJarTp69KgiIyOVnp6uWrVq2bRZt27dNb9/UfycijsBoKDatGmjjh076qWXXtLAgQNtpr3//vuaOnWqpkyZorCwMJUpU0ajRo3Kd6BYnz59NGbMGP3xxx+6evWqTp48aR4Sjo+PlyTNnDkzx1gMR0fHfPMsU6aMatSoYT7/7LPP5OPjo5kzZ+rNN9/U5s2b1bdvX73xxhvq2LGjfHx89NVXX+mDDz4w53FxcVH//v01e/Zs9ejRQ/Pnz9fUqVPN6fHx8br//vs1ceLEHOsvX768HB0dtWrVKm3atEk///yzPvroI73yyivaunWrqlWrlm/+AICic+DAAXO7Gx8fr/Lly+c6XsDX11eS5O7uXuBlh4SE6ODBg1q9erVWrVql4cOH6/3339e6devk7OxcqHyzz2exWOy6slWZMmVsnsfHx+uxxx6zGQuSqXLlytqzZ48cHR21Y8eOHN+3np6ehc4DNweFBUqVd999V40aNVLt2rVt4r/99pu6du2q//u//5OUcS7pX3/9pdtuuy3PZVWqVEnh4eGaN2+erl69qnvuuUflypWTJAUFBalChQo6cuSIeRSjsCwWixwcHHT16lVJ0qZNm1SlShW98sorZpusRzMyDR06VPXr19f06dOVlpamHj16mNMaN26sr7/+WlWrVpWTU+4fY4vFolatWqlVq1Z67bXXVKVKFS1dulTPPvusXf0BABTML7/8osjISD3zzDOSMrbd586dk5OTU56XYm3QoIHWrFmjQYMGFWgd7u7uuv/++3X//fdrxIgRqlOnjiIjI9W4cWObdnXr1tXJkyd18uRJ86jF/v37dfny5Xy/K4ta48aNtX//fpsf4LK6/fbblZ6erqioKN111103LS8UDQoLlCphYWHq27evPvzwQ5t4zZo1tWTJEm3atEl+fn6aNGmSzp8/f82NZd++fTVu3DilpKSYg6czvfHGG3rqqafk4+Oje++9V8nJyfr9998VExOT7855cnKyzp07J0mKiYnRtGnTzCMMmbmeOHFCX331lZo1a6YffvjBPOUpq7p16+qOO+7QmDFjNHjwYJtfsUaMGKGZM2eqT58+euGFF+Tv76+///5bX331lT777DP9/vvvWrNmjTp06KBy5cpp69atunDhgurWrZv/CwwAKJTMbX96errOnz+vlStX6p133tF9992n/v37S8q4ymHLli3VrVs3vffee6pVq5bOnDljDthu2rSpxo0bp/bt26t69ep6+OGHlZaWphUrVmjMmDE51jlnzhylp6erRYsW8vDw0P/+9z+5u7urSpUqOdpGRESY36FTpkxRWlqahg8frvDw8BynK91IY8aM0R133KGRI0dq6NChKlOmjPbv369Vq1Zp2rRpqlWrlvr27av+/fvrgw8+0O23364LFy5ozZo1atCggbp06XLTcsX1Y4wFSp3x48fnOCw7duxYNW7cWB07dlTbtm0VHBysbt26XXNZDz74oC5evKjExMQc7YcOHarPPvtMs2fPVlhYmMLDwzVnzpxrnkq0cuVKlS9fXuXLl1eLFi20fft2LV68WG3btpUkPfDAA3rmmWc0cuRINWrUSJs2bdKrr76a67KGDBmilJQUDR482CZeoUIF/fbbb0pPT1eHDh0UFhamUaNGydfXVw4ODvL29tb69evVuXNn1apVS2PHjtUHH3ygTp06XfM1AQBcv8xtf9WqVXXvvffq119/1Ycffqhvv/3WPKXHYrFoxYoVatOmjQYNGqRatWrp4Ycf1vHjxxUUFCQp49Loixcv1nfffadGjRqpXbt2eV7h0NfXVzNnzlSrVq3UoEEDrV69WsuXL7e5CmEmi8Wib7/9Vn5+fmrTpo0iIiIUGhqqhQsX3rgXJRcNGjTQunXr9Ndff+muu+7S7bffrtdee00VKlQw28yePVv9+/fX6NGjVbt2bXXr1k3bt29X5cqVb2quuH4Ww+D6k0BJNWHCBC1evJg7dwMAgBKPIxZACRQfH6+9e/dq2rRpevLJJ4s7HQAAgGuisABKoJEjR6pJkyZq27ZtjtOgAAAASiJOhQIAAABgN45YAAAAALAbhQUAAAAAu1FYAAAAALAbhQUAAAAAu1FYAAAAALAbhQUAlEBVq1bVfffdd812a9eulcVi0dq1a298UreIY8eOyWKxaM6c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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"  PROJECT SUMMARY\")\n",
        "\n",
        "print(f\"  Dataset         : Food Ingredients (Kaggle)\")\n",
        "print(f\"  Total Samples   : {df.shape[0]:,}\")\n",
        "print(f\"  Training Set    : {X_train.shape[0]:,} samples\")\n",
        "print(f\"  Testing Set     : {X_test.shape[0]:,} samples\")\n",
        "print(f\"  Features (TF-IDF): 5,000 word features\")\n",
        "print(f\"\")\n",
        "print(f\"  Naive Bayes     : {nb_accuracy * 100:.2f}%\")\n",
        "print(f\"  Decision Tree   : {dt_accuracy * 100:.2f}%\")\n",
        "print(f\"\")\n",
        "print(f\"    Best Model   : {winner_name}\")\n",
        "print(f\"    Accuracy     : {winner_acc * 100:.2f}%\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ic8XaWluA7qX",
        "outputId": "dbc6780b-f336-4161-f4ed-f491afca418a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  PROJECT SUMMARY\n",
            "  Dataset         : Food Ingredients (Kaggle)\n",
            "  Total Samples   : 39,787\n",
            "  Training Set    : 31,829 samples\n",
            "  Testing Set     : 7,958 samples\n",
            "  Features (TF-IDF): 5,000 word features\n",
            "\n",
            "  Naive Bayes     : 86.53%\n",
            "  Decision Tree   : 99.18%\n",
            "\n",
            "    Best Model   : Decision Tree\n",
            "    Accuracy     : 99.18%\n"
          ]
        }
      ]
    }
  ]
}