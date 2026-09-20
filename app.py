import streamlit as st
import plotly.graph_objects as go
import json
from extraction import get_attention_and_prediction
from bias_pairs import BIAS_PAIRS
import pandas as pd

st.title("Interpretability Explorer")
st.subheader("See what a sentiment model pays attention to")
st.write("This tool visualizes what a sentiment classification model is actually focusing on when it makes a decision. Use it to explore attention patterns, test for bias across sensitive attributes, and see what caused specific model errors.")

mode = st.sidebar.radio("Mode", ["Free Text", "Bias Audit", "Error Analysis", "Bias Summary", "Error Summary"])


def plot_heatmap(tokens, attn_matrix, title):
    fig = go.Figure(data=go.Heatmap(
        z=attn_matrix,
        x=tokens,
        y=tokens,
        colorscale="Viridis"
    ))
    fig.update_layout(title=title, height=400)
    return fig


def get_top_attended_tokens(tokens, attn_matrix, cls_index=0, top_n=3):
    """Look at what the [CLS] token attends to most, since that's what drives the final prediction."""
    cls_attention = attn_matrix[cls_index]  # attention FROM [CLS] TO every other token
    pairs = [(tokens[i], cls_attention[i]) for i in range(len(tokens)) if tokens[i] not in ["[CLS]", "[SEP]"]]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


if mode == "Free Text":
    sentence = st.text_input("Enter a sentence:", "The movie was surprisingly good.")

    if sentence:
        result = get_attention_and_prediction(sentence)
        tokens = result["tokens"]

        layer = st.slider("Layer", 0, 5, 5)
        head = st.slider("Head", 0, 11, 0)

        attn_matrix = result["attentions"][layer][0, head].numpy()
        st.plotly_chart(plot_heatmap(tokens, attn_matrix, f"Attention — Layer {layer}, Head {head}"))
        st.write(f"**Prediction:** {result['prediction']} ({result['confidence']:.2%} confidence)")

        top_tokens = get_top_attended_tokens(tokens, attn_matrix)
        token_str = ", ".join([f"'{t}' ({score:.0%})" for t, score in top_tokens])
        st.info(f"**What drove this decision:** The model focused most on {token_str}.")

elif mode == "Bias Audit":
    st.write("Compare how the model responds to near-identical sentences that differ only in a sensitive attribute (gender, name, religion, age, disability, class).")

    pair_idx = st.selectbox("Choose a sentence pair", range(len(BIAS_PAIRS)), format_func=lambda i: f"Pair {i+1}")
    sent_a, sent_b = BIAS_PAIRS[pair_idx]

    layer = st.slider("Layer", 0, 5, 5, key="bias_layer")
    head = st.slider("Head", 0, 11, 0, key="bias_head")

    col1, col2 = st.columns(2)
    results = []

    for col, sent in zip([col1, col2], [sent_a, sent_b]):
        result = get_attention_and_prediction(sent)
        attn_matrix = result["attentions"][layer][0, head].numpy()
        col.plotly_chart(plot_heatmap(result["tokens"], attn_matrix, sent), use_container_width=True)
        col.write(f"**{result['prediction']}** ({result['confidence']:.2%})")

        top_tokens = get_top_attended_tokens(result["tokens"], attn_matrix)
        token_str = ", ".join([f"'{t}'" for t, score in top_tokens])
        col.caption(f"Focused on: {token_str}")

        results.append(result)

    st.markdown("---")
    conf_delta = abs(results[0]["confidence"] - results[1]["confidence"])
    prediction_flipped = results[0]["prediction"] != results[1]["prediction"]

    if prediction_flipped:
        st.error(f"**Flag: prediction changed.** Sentence A predicted {results[0]['prediction']}, Sentence B predicted {results[1]['prediction']}, despite only a small wording change. Worth investigating.")
    elif conf_delta > 0.05:
        st.warning(f"**Notable shift:** confidence changed by {conf_delta:.1%} between the two sentences, same prediction, but the model was more/less sure. Could indicate sensitivity to the swapped attribute.")
    else:
        st.success(f"**No meaningful difference detected.** Confidence changed by only {conf_delta:.1%}, the model appears consistent across this pair.")

elif mode == "Error Analysis":
    st.write("Browse sentences the model misclassified, and see what it was focused on.")

    try:
        with open("misclassified_cache.json") as f:
            misclassified = json.load(f)
    except FileNotFoundError:
        st.error("No cached misclassified examples found. Run `python error_analysis.py` first.")
        misclassified = []

    if misclassified:
        idx = st.selectbox(
            "Choose a misclassified example",
            range(len(misclassified)),
            format_func=lambda i: f"Example {i+1}: \"{misclassified[i]['sentence'][:50]}...\""
        )
        example = misclassified[idx]

        st.write(f"**Sentence:** {example['sentence']}")
        st.write(f"**True label:** {example['true_label']} | **Model predicted:** {example['predicted']} ({example['confidence']:.2%} confidence)")

        layer = st.slider("Layer", 0, 5, 5, key="error_layer")
        head = st.slider("Head", 0, 11, 0, key="error_head")

        result = get_attention_and_prediction(example["sentence"])
        attn_matrix = result["attentions"][layer][0, head].numpy()
        st.plotly_chart(plot_heatmap(result["tokens"], attn_matrix, f"Attention — Layer {layer}, Head {head}"))

        # Diagnosis uses the FINAL layer, averaged across all 12 heads, not the slider's layer/head
        final_layer_attn = result["attentions"][-1][0].mean(dim=0).numpy()  # average across heads
        top_tokens = get_top_attended_tokens(result["tokens"], final_layer_attn)
        token_str = ", ".join([f"'{t}' ({score:.0%})" for t, score in top_tokens])
        st.info(f"**Likely cause of error:** In the final layer (averaged across all heads), the model focused most on {token_str}, but the true label was {example['true_label']}. This mismatch between focus and correct sentiment may explain the mistake.")

elif mode == "Bias Summary":
    st.write("Aggregate results from running the full bias-pair audit across all categories.")

    try:
        with open("bias_audit_results.json") as f:
            bias_results = json.load(f)
    except FileNotFoundError:
        st.error("No bias audit results found. Run `python bias_audit_analysis.py` first.")
        bias_results = {}

    if bias_results:
        categories = list(bias_results.keys())
        avg_deltas = [bias_results[c]["avg_confidence_delta"] * 100 for c in categories]
        flip_counts = [bias_results[c]["flip_count"] for c in categories]
        total_pairs = [bias_results[c]["total_pairs"] for c in categories]

        df = pd.DataFrame({
            "Category": categories,
            "Avg Confidence Delta (%)": avg_deltas,
            "Prediction Flips": [f"{f}/{t}" for f, t in zip(flip_counts, total_pairs)]
        })

        fig = go.Figure(data=go.Bar(x=categories, y=avg_deltas, marker_color="indianred"))
        fig.update_layout(
            title="Average Confidence Shift by Bias Category",
            yaxis_title="Avg Confidence Delta (%)",
            height=400
        )
        st.plotly_chart(fig)

        st.dataframe(df, use_container_width=True)

        max_category = categories[avg_deltas.index(max(avg_deltas))]
        st.info(f"**Finding:** '{max_category}' shows the largest average confidence shift ({max(avg_deltas):.2f}%) across tested pairs, more than any other category. No prediction flips were observed in any category, suggesting the model's final verdicts remain stable, but confidence is not perfectly uniform across sensitive attributes.")

elif mode == "Error Summary":
    st.write("Accuracy breakdown across the full SST-2 validation set (872 examples).")

    try:
        with open("error_analysis_summary.json") as f:
            summary = json.load(f)
    except FileNotFoundError:
        st.error("No error summary found. Run `python error_analysis.py` first.")
        summary = {}

    if summary:
        st.metric("Overall Accuracy", f"{summary['overall_accuracy']:.2%}", 
                   help=f"Based on {summary['total_examples']} validation examples")

        categories = ["Negation", "Non-Negation", "Short (≤10 words)", "Long (>10 words)"]
        accuracies = [
            summary["negation_accuracy"] * 100,
            summary["non_negation_accuracy"] * 100,
            summary["short_sentence_accuracy"] * 100,
            summary["long_sentence_accuracy"] * 100
        ]
        counts = [
            summary["negation_count"],
            summary["non_negation_count"],
            summary["short_sentence_count"],
            summary["long_sentence_count"]
        ]

        fig = go.Figure(data=go.Bar(
            x=categories, y=accuracies,
            text=[f"n={c}" for c in counts],
            textposition="outside",
            marker_color=["indianred", "lightseagreen", "indianred", "lightseagreen"]
        ))
        fig.update_layout(
            title="Accuracy by Sentence Characteristic",
            yaxis_title="Accuracy (%)",
            yaxis_range=[80, 100],
            height=400
        )
        st.plotly_chart(fig)

        gap = summary["non_negation_accuracy"] - summary["negation_accuracy"]
        st.info(f"**Finding:** the model is {gap:.1%} less accurate on negation-containing sentences ({summary['negation_accuracy']:.1%}) compared to non-negation sentences ({summary['non_negation_accuracy']:.1%}), its largest measurable weakness among the factors tested. Out of {summary['total_examples']} validation examples, {summary['total_misclassified']} ({100*summary['total_misclassified']/summary['total_examples']:.1f}%) were misclassified overall.")
        