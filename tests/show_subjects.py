"""Show all subjects, topics, and data files. Run: python -m tests.show_subjects"""
import sys, json, os
sys.path.insert(0, '.')

domains = {
    'python_programming':  ('data/prerequisite_graph.json',      'data/corpus/python_programming.md',  'data/pretest_python.json',            'data/posttest_python.json'),
    'ml_basics':           ('data/ml_basics_graph.json',         'data/corpus/ml_basics.md',            'data/pretest_ml_basics.json',         'data/posttest_ml_basics.json'),
    'signal_processing':   ('data/signal_processing_graph.json', 'data/corpus/signal_processing.md',   'data/pretest_signal_processing.json', 'data/posttest_signal_processing.json'),
    'physiology_basics':   ('data/physiology_graph.json',        'data/corpus/physiology_basics.md',   'data/pretest_physiology_basics.json', 'data/posttest_physiology_basics.json'),
    'data_analysis':       ('data/data_analysis_graph.json',     'data/corpus/data_analysis.md',       'data/pretest_data_analysis.json',     'data/posttest_data_analysis.json'),
}

display_names = {
    'python_programming': 'Python Programming',
    'ml_basics':          'Basics of ML',
    'signal_processing':  'Signal Processing',
    'physiology_basics':  'Physiology Basics',
    'data_analysis':      'Data Analysis',
}

print()
print('=' * 65)
print('  YOUR PROJECT — 5 SUBJECTS')
print('=' * 65)

for domain, (graph_file, corpus_file, pre_file, post_file) in domains.items():
    with open(graph_file) as f:
        topics = json.load(f)['topics']

    corpus_kb   = round(os.path.getsize(corpus_file) / 1024, 1) if os.path.exists(corpus_file) else 0
    pre_q       = len(json.load(open(pre_file))['questions'])   if os.path.exists(pre_file)    else 0
    post_q      = len(json.load(open(post_file))['questions'])  if os.path.exists(post_file)   else 0

    print()
    print('  ' + display_names[domain] + ' (' + domain + ')')
    print('  ' + '-' * 60)
    print('  Topics (' + str(len(topics)) + '):')
    for i, (t, info) in enumerate(topics.items(), 1):
        prereqs = info.get('prerequisites', [])
        bloom   = info.get('bloom_level', 'N/A')
        obj     = info.get('objective', '')[:55]
        pre_str = ' ← needs: ' + ', '.join(prereqs) if prereqs else ' (starting point)'
        print('    ' + str(i).rjust(2) + '. ' + t)
        print('        objective: ' + obj + ('...' if len(info.get('objective','')) > 55 else ''))
        print('        bloom: ' + bloom + pre_str)

    print()
    print('  Data files:')
    print('    Corpus (course notes): ' + corpus_file + ' (' + str(corpus_kb) + ' KB)' + ('' if corpus_kb else '  MISSING!'))
    print('    Pre-test:  ' + pre_file  + ' (' + str(pre_q)  + ' questions)' + ('' if pre_q  else '  MISSING!'))
    print('    Post-test: ' + post_file + ' (' + str(post_q) + ' questions)' + ('' if post_q else '  MISSING!'))

print()
print('=' * 65)
print('  SUMMARY')
print('=' * 65)
total_topics = 0
for domain, (graph_file, *_) in domains.items():
    n = len(json.load(open(graph_file))['topics'])
    total_topics += n
    print('  ' + display_names[domain].ljust(25) + str(n) + ' topics')
print('  ' + '-' * 35)
print('  Total'.ljust(25) + str(total_topics) + ' topics across 5 subjects')
print()
print('  LLM (Groq gpt-oss-120b) is called for:')
print('    - Planner:    personalise topic order per student')
print('    - Content:    generate explanation + example per topic')
print('    - Assessment: generate MCQ question per topic')
print('    - Assessment: grade short-answer responses')
print()
print('  Pre/post tests: FIXED (same 10 Qs for all students)')
print('  In-session Qs:  MODEL-GENERATED (different every time)')
print('=' * 65)
