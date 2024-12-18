for i in {1..5}; do
    for desp_file in $(ls -p results/q_pathway/ | grep -v '/$' | grep -v 'q_pathway_eval.json' | grep -v 'completeness_scores.*\.json'); do
        python eval/reason/description/gpt_eval/cal_gpt_score_completeness.py --desp_file "results/q_pathway/$desp_file"
    done
done
# for i in {1..5}; do
#     for desp_file in $(ls -p results/q_pathway/ | grep -v '/$' | grep -v 'q_pathway_eval.json' | grep -v 'relevance_scores.*\.json'); do
#         python eval/reason/description/gpt_eval/cal_gpt_score_relevance.py --desp_file "results/q_pathway/$desp_file"
#     done
# done
# for i in {1..5}; do
#     for desp_file in $(ls -p results/q_pathway/ | grep -v '/$' | grep -v 'q_pathway_eval.json' | grep -v 'preciseness_scores.*\.json'); do
#         python eval/reason/description/gpt_eval/cal_gpt_score_preciseness.py --desp_file "results/q_pathway/$desp_file"
#     done
# done
