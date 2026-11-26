# Buddhist Eval

Scores the output of AI models (OCR, speech-to-text, span annotation) against human-made ground truth, with first-class support for Tibetan.

## Text scoring

**Reference**:
The ground-truth text a human produced for one item.
_Avoid_: Label, gold text, target

**Prediction**:
The text a model produced for the same item as a Reference.
_Avoid_: Hypothesis, inference, output, submission

**CER** (Character Error Rate):
Character edits (substitutions + deletions + insertions) needed to turn the Prediction into the Reference, divided by the Reference's character count.

**WER** (Word Error Rate):
The same ratio as CER, counted over words instead of characters. For Tibetan, a word is a syllable, because the tsek becomes the word boundary.
_Avoid_: Syllable error rate (for Tibetan, it is the same thing; use WER)

**Language**:
The language of a Reference and Prediction pair, which decides how text is prepared before it is scored (for Tibetan, conversion to Wylie).

## Span scoring

**Span**:
A labelled, contiguous range of character offsets in a text.
_Avoid_: Annotation, entity, segment

**Gold Span**:
A Span from the ground truth.
_Avoid_: Ground-truth span, reference span

**Predicted Span**:
A Span produced by a model.
_Avoid_: Inference span, hypothesis span

**Tag**:
The label carried by a Span, for example a quotation class. A Predicted Span can only match a Gold Span with the same Tag.
_Avoid_: Class, category, type

**Match**:
A one-to-one pairing of a Predicted Span and a Gold Span whose IoU meets the IoU Threshold.

**IoU Threshold**:
The minimum overlap (intersection over union) at which a Predicted Span and a Gold Span count as a Match.

## TOC scoring

**Heading**:
An outline heading (ས་བཅད, sa-bcad) in a passage, with a depth (1 is the shallowest level in that passage) and a start offset. A Gold Heading comes from the ground truth.
_Avoid_: Section, chapter, TOC entry

**Anchor**:
The verbatim text a Predicted Heading starts with. It is located in the passage to give the heading a start offset.

**Tolerance**:
The maximum character distance between a located Anchor and a Gold Heading's start for the two to match.

## MCQ scoring

**Answer Key**:
The correct choice for each multiple-choice question, keyed by question id.
_Avoid_: Gold answers, labels

**Accuracy**:
The share of Answer Key questions whose predicted choice equals the correct choice, ignoring case and surrounding whitespace. A question with no prediction is wrong.
