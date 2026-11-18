"""Contrast pairs checked against exact text from the QASPER development corpus."""

# Each row includes the paper, verified source anchor, supported claim and edited claim.
PAIRS = [
    ("1810.08699", "53453 tokens and 2566 sentences", "The manually annotated test dataset contains 53,453 tokens.", "The manually annotated test dataset contains 35,453 tokens.", "CONTRADICTED"),
    ("1810.08699", "53453 tokens and 2566 sentences", "The manually annotated test dataset contains 2,566 sentences.", "The manually annotated test dataset contains 5,266 sentences.", "CONTRADICTED"),
    ("1810.08699", "over 250 news texts from ilur.am", "The manually annotated test sentences were selected from news texts from ilur.am.", "The manually annotated test sentences were selected from news texts from Reuters.", "CONTRADICTED"),
    ("1810.08699", "between August 2012 and July 2018", "The manually annotated test news covers August 2012 to July 2018.", "The manually annotated test news covers August 2014 to July 2018.", "CONTRADICTED"),
    ("1810.08699", "people (PER), organizations (ORG), and locations (LOC)", "The pioNER test set labels people, organizations and locations.", "The pioNER test set labels people, organizations and dates as its three entity classes.", "CONTRADICTED"),
    ("1810.08699", "163000-token named entity corpus automatically generated", "The manually annotated news and automatically generated Wikipedia corpora are for Armenian.", "Both corpus creation procedures would achieve equally accurate results for Georgian.", "INSUFFICIENT"),
    ("1609.00425", "total training dataset of 5000 posts", "The Reddit dogmatism training dataset contains 5,000 posts.", "The Reddit dogmatism training dataset contains 50,000 posts.", "CONTRADICTED"),
    ("1609.00425", "1000 additional posts from the Reddit frontpage", "The initial training sample includes 1,000 posts from the Reddit frontpage.", "The initial training sample includes 2,000 posts from the Reddit frontpage.", "CONTRADICTED"),
    ("1609.00425", "300 and 400 characters", "The Reddit training posts have lengths between 300 and 400 characters.", "The Reddit training posts have lengths between 300 and 400 words.", "CONTRADICTED"),
    ("1609.00425", "between January 2007 and March 2015", "The original Reddit training sample covers January 2007 to March 2015.", "The original Reddit training sample covers January 2007 to March 2019.", "CONTRADICTED"),
    ("1609.00425", "L2 penalties of 1.5", "The BOW classifiers use an L2 penalty of 1.5.", "The BOW classifiers use an L2 penalty of 0.15.", "CONTRADICTED"),
    ("1609.00425", "annotated via paid crowdsourcing", "The model's Reddit corpus was annotated via paid crowdsourcing.", "The Reddit dogmatism classifier works equally well on every major social media platform.", "INSUFFICIENT"),
    ("1909.09067", "approximately 211,000 sentences", "The German readability and simplification corpus has about 211,000 sentences.", "The German readability and simplification corpus has about 21,000 sentences.", "CONTRADICTED"),
    ("1909.09067", "information on text structure, typography, and images", "The German readability corpus contains structural, typographic and image information.", "The image information improves every readability model for all languages.", "INSUFFICIENT"),
    ("1909.09067", "font type and font style", "The secondary data specifies token font type and font style.", "Token font style improves test accuracy by exactly 10 percentage points on every dataset.", "INSUFFICIENT"),
    ("1704.06194", "single-relation (SimpleQuestions) and multi-relation (WebQSP)", "The paper reports state-of-the-art KBQA accuracy on SimpleQuestions and WebQSP.", "The paper demonstrates state-of-the-art KBQA accuracy on every single- and multi-relation benchmark.", "INSUFFICIENT"),
    ("1704.06194", "uses deep residual bidirectional LSTMs", "The relation detector uses deep residual bidirectional LSTMs.", "The relation detector uses bidirectional LSTMs that yield the same improvement in every language.", "INSUFFICIENT"),
    ("1909.00512", "less than 5% of the variance", "For ELMo, BERT and GPT-2, a static embedding explains less than 5% of contextualized variance on average across the examined layers.", "For ELMo, BERT and GPT-2, a static embedding explains more than 50% of contextualized variance on average across the examined layers.", "CONTRADICTED"),
    ("1909.00512", "this self-similarity is much lower in upper layers", "The self-similarity of word representations is lower in upper layers of the examined models.", "The upper-layer decrease in self-similarity is proven for every contextual language model.", "INSUFFICIENT"),
    ("1801.05147", "Chinese Named Entity Recognition (NER)", "The proposed crowd annotation learning approach is applied to Chinese named entity recognition.", "The same approach has been shown to retain identical performance on all languages and entity types.", "INSUFFICIENT"),
]
