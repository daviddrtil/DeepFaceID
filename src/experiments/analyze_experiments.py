"""
Analyze experiment results from stats.txt files.
Parses output directories to evaluate deepfake detection accuracy.

Usage: python src/analyze_experiments.py [--outputs-dir PATH]
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict

DEEPFAKE_SCORE_THRESHOLD = 0.50


def parse_folder_name(folder_name):
    """Extract session_name, label, and timestamp from folder name."""
    parts = folder_name.rsplit('_', 3)
    if len(parts) >= 4:
        date_part = parts[-2]
        time_part = parts[-1]
        if re.match(r'\d{4}-\d{2}-\d{2}', date_part) and re.match(r'\d{2}-\d{2}-\d{2}', time_part):
            label_part = parts[-3]
            if label_part in ('real', 'fake'):
                session_name = '_'.join(parts[:-3])
                return session_name, label_part, f"{date_part}_{time_part}"
            else:
                session_name = '_'.join(parts[:-2])
                return session_name, None, f"{date_part}_{time_part}"
    
    parts = folder_name.rsplit('_', 2)
    if len(parts) >= 3:
        date_part = parts[-2]
        time_part = parts[-1]
        if re.match(r'\d{4}-\d{2}-\d{2}', date_part) and re.match(r'\d{2}-\d{2}-\d{2}', time_part):
            session_name = '_'.join(parts[:-2])
            return session_name, None, f"{date_part}_{time_part}"
    
    return folder_name, None, None


def parse_stats_line(line):
    """Parse a single line from stats.txt."""
    result = {}
    
    frame_match = re.search(r'frame=(\d+)', line)
    if frame_match:
        result['frame'] = int(frame_match.group(1))
    
    passive_avg_match = re.search(r'passive_avg=(\d+\.\d+|None)', line)
    if passive_avg_match:
        val = passive_avg_match.group(1)
        result['passive_avg'] = None if val == 'None' else float(val)
    
    passive_cur_match = re.search(r'passive_cur=(\d+\.\d+|None)', line)
    if passive_cur_match:
        val = passive_cur_match.group(1)
        result['passive_cur'] = None if val == 'None' else float(val)
    
    spatial_match = re.search(r'spatial=(\d+\.\d+|None)', line)
    if spatial_match:
        val = spatial_match.group(1)
        result['spatial'] = None if val == 'None' else float(val)
    
    spatial_frame_match = re.search(r'spatial_frame=(\d+)', line)
    if spatial_frame_match:
        result['spatial_frame'] = int(spatial_frame_match.group(1))
    
    action_match = re.search(r'action=([^\s|]+)', line)
    if action_match:
        action_val = action_match.group(1).strip()
        result['action'] = None if action_val == 'None' else action_val
    
    face_match = re.search(r'face=(\d)', line)
    if face_match:
        result['face_detected'] = int(face_match.group(1)) == 1
    
    hand_match = re.search(r'hand=(\d)', line)
    if hand_match:
        result['hand_detected'] = int(hand_match.group(1)) == 1
    
    return result


def load_session(stats_file):
    """Load all frames and summary from a stats.txt file."""
    frames = []
    summary = {}
    in_summary = False
    with open(stats_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('--- SUMMARY'):
                in_summary = True
                continue
            if in_summary:
                if line.startswith('label='):
                    summary['label'] = line.split('=', 1)[1]
                elif line.startswith('final_decision='):
                    summary['final_decision'] = line.split('=', 1)[1]
            else:
                parsed = parse_stats_line(line)
                if parsed:
                    frames.append(parsed)
    return frames, summary


def analyze_session(frames, ground_truth):
    """Analyze a single session's frames."""
    if not frames:
        return None
    
    last_frame = frames[-1]
    final_passive_avg = last_frame.get('passive_avg')
    
    if final_passive_avg is None:
        return None
    
    passive_predicts_real = final_passive_avg <= DEEPFAKE_SCORE_THRESHOLD
    is_actually_real = ground_truth == 'real'
    passive_correct = passive_predicts_real == is_actually_real
    
    actions_seen = set()
    action_frames = defaultdict(list)
    for frame in frames:
        action = frame.get('action')
        if action and action != 'None':
            actions_seen.add(action)
            if frame.get('passive_avg') is not None:
                action_frames[action].append(frame)
    
    return {
        'ground_truth': ground_truth,
        'final_passive_avg': final_passive_avg,
        'passive_predicts_real': passive_predicts_real,
        'passive_correct': passive_correct,
        'actions_seen': actions_seen,
        'action_frames': action_frames,
        'total_frames': len(frames),
    }


def find_sessions(outputs_dir):
    """Find all session directories with stats.txt files."""
    sessions = []
    outputs_path = Path(outputs_dir)
    
    if not outputs_path.exists():
        return sessions
    
    for folder in outputs_path.iterdir():
        if not folder.is_dir():
            continue
        
        stats_file = folder / 'stats.txt'
        if not stats_file.exists():
            continue
        
        session_name, label, timestamp = parse_folder_name(folder.name)
        sessions.append({
            'folder': folder,
            'stats_file': stats_file,
            'session_name': session_name,
            'label': label,
            'timestamp': timestamp,
        })
    
    return sorted(sessions, key=lambda x: x['timestamp'] or '')


def calculate_metrics(results):
    """Calculate accuracy metrics from results."""
    if not results:
        return {}
    
    total = len(results)
    correct = sum(1 for r in results if r['passive_correct'])
    
    real_results = [r for r in results if r['ground_truth'] == 'real']
    fake_results = [r for r in results if r['ground_truth'] == 'fake']
    
    tp = sum(1 for r in real_results if r['passive_predicts_real'])
    fn = sum(1 for r in real_results if not r['passive_predicts_real'])
    fp = sum(1 for r in fake_results if r['passive_predicts_real'])
    tn = sum(1 for r in fake_results if not r['passive_predicts_real'])
    
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'real_count': len(real_results),
        'fake_count': len(fake_results),
    }


def calculate_action_metrics(results):
    """Calculate accuracy metrics per action type."""
    action_results = defaultdict(lambda: {'real_scores': [], 'fake_scores': []})
    
    for r in results:
        gt = r['ground_truth']
        for action, frames in r['action_frames'].items():
            scores = [f['passive_avg'] for f in frames if f.get('passive_avg') is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                if gt == 'real':
                    action_results[action]['real_scores'].append(avg_score)
                else:
                    action_results[action]['fake_scores'].append(avg_score)
    
    metrics = {}
    for action, data in action_results.items():
        real_correct = sum(1 for s in data['real_scores'] if s <= DEEPFAKE_SCORE_THRESHOLD)
        fake_correct = sum(1 for s in data['fake_scores'] if s > DEEPFAKE_SCORE_THRESHOLD)
        total = len(data['real_scores']) + len(data['fake_scores'])
        correct = real_correct + fake_correct
        
        metrics[action] = {
            'accuracy': correct / total if total > 0 else 0,
            'total': total,
            'correct': correct,
            'real_count': len(data['real_scores']),
            'fake_count': len(data['fake_scores']),
            'avg_real_score': sum(data['real_scores']) / len(data['real_scores']) if data['real_scores'] else None,
            'avg_fake_score': sum(data['fake_scores']) / len(data['fake_scores']) if data['fake_scores'] else None,
        }
    
    return metrics


def print_report(metrics, action_metrics, results):
    """Print analysis report."""
    print("\n" + "=" * 70)
    print("DEEPFAKE DETECTION ANALYSIS REPORT")
    print("=" * 70)
    
    if not metrics:
        print("\nNo labeled sessions found for analysis.")
        print("Label your sessions using the web UI checkbox.")
        return
    
    print(f"\nSessions analyzed: {metrics['total']}")
    print(f"  Real sessions: {metrics['real_count']}")
    print(f"  Fake sessions: {metrics['fake_count']}")
    
    print("\n" + "-" * 70)
    print("OVERALL METRICS (Passive Analysis)")
    print("-" * 70)
    print(f"  Accuracy:  {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1 Score:  {metrics['f1']:.1%}")
    
    print("\n  Confusion Matrix:")
    print(f"                    Predicted REAL | Predicted FAKE")
    print(f"    Actual REAL:    {metrics['tp']:4d}           | {metrics['fn']:4d}")
    print(f"    Actual FAKE:    {metrics['fp']:4d}           | {metrics['tn']:4d}")
    
    if action_metrics:
        print("\n" + "-" * 70)
        print("ACCURACY PER ACTION TYPE")
        print("-" * 70)
        
        sorted_actions = sorted(action_metrics.items(), key=lambda x: -x[1]['accuracy'])
        for action, m in sorted_actions:
            if m['total'] < 2:
                continue
            print(f"  {action:30s} {m['accuracy']:5.1%}  (n={m['total']})")
            if m['avg_real_score'] is not None:
                print(f"    avg score (real): {m['avg_real_score']:.3f}")
            if m['avg_fake_score'] is not None:
                print(f"    avg score (fake): {m['avg_fake_score']:.3f}")
    
    print("\n" + "-" * 70)
    print("RECENT SESSIONS")
    print("-" * 70)
    for r in results[-10:]:
        status = "✓" if r['passive_correct'] else "✗"
        gt = r['ground_truth'].upper()
        pred = "REAL" if r['passive_predicts_real'] else "FAKE"
        score = r['final_passive_avg']
        print(f"  {status} GT:{gt:4s} Pred:{pred:4s} Score:{score:.3f}")
    
    print("\n" + "=" * 70 + "\n")


def export_csv(results, output_file):
    """Export results to CSV."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("session,ground_truth,final_passive_avg,predicted,correct,total_frames\n")
        for r in results:
            pred = 'real' if r['passive_predicts_real'] else 'fake'
            correct = 'yes' if r['passive_correct'] else 'no'
            f.write(f"{r.get('session_name','')},{r['ground_truth']},{r['final_passive_avg']:.4f},{pred},{correct},{r['total_frames']}\n")
    print(f"Exported {len(results)} sessions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze deepfake detection experiments")
    parser.add_argument("--outputs-dir", type=str, default=None, help="Path to outputs directory")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else project_root / "outputs"
    
    print(f"Scanning: {outputs_dir}")
    sessions = find_sessions(outputs_dir)
    print(f"Found {len(sessions)} sessions with stats.txt")
    
    results = []
    for session in sessions:
        frames, summary = load_session(session['stats_file'])
        # Prefer label from summary, fallback to folder name
        label = summary.get('label') if summary.get('label') not in (None, 'unknown') else session['label']
        if label not in ('real', 'fake'):
            continue
        analysis = analyze_session(frames, label)
        if analysis:
            analysis['session_name'] = session['session_name']
            analysis['timestamp'] = session['timestamp']
            analysis['final_decision'] = summary.get('final_decision', 'unknown')
            results.append(analysis)
    
    print(f"Labeled sessions: {len(results)}")
    
    metrics = calculate_metrics(results)
    action_metrics = calculate_action_metrics(results)
    
    print_report(metrics, action_metrics, results)
    
    print_report(metrics, action_metrics, results)
    
    if results:
        csv_file = outputs_dir / "experiments" / "analysis_results.csv"
        export_csv(results, csv_file)


if __name__ == '__main__':
    main()
