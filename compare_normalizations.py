import os
import sys
from collections import defaultdict

def collect_valid_files(folder):
    valid_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.ann') and not file.endswith('-checkpoint.ann'):
                valid_files.append(os.path.join(root, file))
    return valid_files

def parse_annotations(file, reference_keyword):
    annotations = {}
    normalizations = {}
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('T'):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    ann_id, ann_info, text = parts[0], parts[1], parts[2]
                    ann_type, span_info = ann_info.split(' ', 1)
                    spans = [(int(start), int(end)) for start, end in (span.split() for span in span_info.split(';'))]
                    if spans:
                        start = min(span[0] for span in spans)
                        end = max(span[1] for span in spans)
                        annotations[ann_id] = {'type': ann_type, 'spans': [(start, end)], 'text': text}
            elif line.startswith('N'):
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    norm_id, norm_info, text = parts
                    norm_info_splits = norm_info.split(' ')
                    target, reference = norm_info_splits[1], norm_info_splits[2]
                    if reference_keyword in reference:
                        normalizations[target] = reference                
    return annotations, normalizations

def compare_exact(gold_ann, eval_ann, gold_norms, eval_norms):
    tp, fp, fn = 0, 0, 0

    gold_spans = {(ann['type'], tuple(span), gold_norms.get(ann_id)) for ann_id, ann in gold_ann.items() for span in ann['spans']}
    eval_spans = {(ann['type'], tuple(span), eval_norms.get(ann_id)) for ann_id, ann in eval_ann.items() for span in ann['spans']}

    tp = len(gold_spans & eval_spans)
    fn = len(gold_spans - eval_spans)
    fp = len(eval_spans - gold_spans)

    return tp, fp, fn

def spans_overlap(span1, span2):
    return span1[0] < span2[1] and span1[1] > span2[0]

def compare_overlap(gold_ann, eval_ann, gold_norms, eval_norms):
    tp, fp, fn = 0, 0, 0

    gold_spans = {(ann['type'], tuple(span), gold_norms.get(ann_id)) for ann_id, ann in gold_ann.items() for span in ann['spans']}
    eval_spans = {(ann['type'], tuple(span), eval_norms.get(ann_id)) for ann_id, ann in eval_ann.items() for span in ann['spans']}

    matched_gold = set()
    matched_eval = set()

    for g_type, g_span, g_norm in gold_spans:
        for e_type, e_span, e_norm in eval_spans:
            if g_type == e_type and spans_overlap(g_span, e_span) and g_norm == e_norm:
                matched_gold.add((g_type, g_span, g_norm))
                matched_eval.add((e_type, e_span, e_norm))

    tp = len(matched_gold)
    fn = len(gold_spans - matched_gold)
    fp = len(eval_spans - matched_eval)

    return tp, fp, fn

def evaluate(gold_folder, eval_folder, reference_keyword):
    gold_files = collect_valid_files(gold_folder)
    eval_files = collect_valid_files(eval_folder)

    all_tp_exact, all_fp_exact, all_fn_exact = 0, 0, 0
    all_tp_overlap, all_fp_overlap, all_fn_overlap = 0, 0, 0

    for gold_file in gold_files:
        base_name = os.path.basename(gold_file)
        eval_file = os.path.join(eval_folder, base_name)
        
        gold_ann, gold_norms = parse_annotations(gold_file, reference_keyword)
        
        if not os.path.exists(eval_file):
            print(f"Missing evaluation file for: {base_name}")
            tp_exact, fp_exact, fn_exact = 0, 0, len(gold_ann)
            tp_overlap, fp_overlap, fn_overlap = 0, 0, len(gold_ann)
        else:
            eval_ann, eval_norms = parse_annotations(eval_file, reference_keyword)
            tp_exact, fp_exact, fn_exact = compare_exact(gold_ann, eval_ann, gold_norms, eval_norms)
            tp_overlap, fp_overlap, fn_overlap = compare_overlap(gold_ann, eval_ann, gold_norms, eval_norms)

        all_tp_exact += tp_exact
        all_fp_exact += fp_exact
        all_fn_exact += fn_exact

        all_tp_overlap += tp_overlap
        all_fp_overlap += fp_overlap
        all_fn_overlap += fn_overlap

        #print(f"{base_name} Exact: TP={tp_exact}, FP={fp_exact}, FN={fn_exact}")
        #print(f"{base_name} Overlap: TP={tp_overlap}, FP={fp_overlap}, FN={fn_overlap}")

    precision_exact = all_tp_exact / (all_tp_exact + all_fp_exact) if all_tp_exact + all_fp_exact > 0 else 0
    recall_exact = all_tp_exact / (all_tp_exact + all_fn_exact) if all_tp_exact + all_fn_exact > 0 else 0
    f1_exact = 2 * precision_exact * recall_exact / (precision_exact + recall_exact) if precision_exact + recall_exact > 0 else 0

    precision_overlap = all_tp_overlap / (all_tp_overlap + all_fp_overlap) if all_tp_overlap + all_fp_overlap > 0 else 0
    recall_overlap = all_tp_overlap / (all_tp_overlap + all_fn_overlap) if all_tp_overlap + all_fn_overlap > 0 else 0
    f1_overlap = 2 * precision_overlap * recall_overlap / (precision_overlap + recall_overlap) if precision_overlap + recall_overlap > 0 else 0

    print("\nSummary (Exact):")
    print(f"TP: {all_tp_exact}, FP: {all_fp_exact}, FN: {all_fn_exact}")
    print(f"Precision: {precision_exact:.4f}")
    print(f"Recall: {recall_exact:.4f}")
    print(f"F1-Score: {f1_exact:.4f}")

    print("\nSummary (Overlap):")
    print(f"TP: {all_tp_overlap}, FP: {all_fp_overlap}, FN: {all_fn_overlap}")
    print(f"Precision: {precision_overlap:.4f}")
    print(f"Recall: {recall_overlap:.4f}")
    print(f"F1-Score: {f1_overlap:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_normalizations.py <evaluation_folder> <gold_folder> \"meddra_pt_id\"")
        sys.exit(1)

    eval_folder = sys.argv[1]
    gold_folder = sys.argv[2]
    reference_keyword = sys.argv[3]

    evaluate(gold_folder, eval_folder, reference_keyword)
