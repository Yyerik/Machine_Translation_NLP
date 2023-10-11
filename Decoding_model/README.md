There are two python programs here (-h for usage) and one txt file:

- `decode` translates input sentences from French to English using beam search with reordering.
- `decode-ext` translates input sentences from French to English using A* search
- `translations` is our best translations obtained by running decode-ext 

These commands work in a pipeline. For example:

    > python decode | python compute-model-score
    > python decode-ext | python compute-model-score

The translations.txt is obtained by running following code: 
    > python decode-ext > translations
