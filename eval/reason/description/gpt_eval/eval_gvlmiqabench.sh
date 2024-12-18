# for i in {1..5}; do
#     for desp_file in $(ls -p results/gvlmiqa_bench/description | grep -v '/$' | grep -v 'benchmark_2k.json' | grep -v 'completeness_scores.*\.json'); do
#         python eval/reason/description/gpt_eval/cal_gpt_score_completeness_gvlmiqa.py --desp_file "results/gvlmiqa_bench/description/$desp_file"
#     done
# done
# for i in {1..5}; do
#     for desp_file in $(ls -p results/gvlmiqa_bench/description | grep -v '/$' | grep -v 'benchmark_2k.json' | grep -v 'relevance_scores.*\.json'); do
#         python eval/reason/description/gpt_eval/cal_gpt_score_relevance_gvlmiqa.py --desp_file "results/gvlmiqa_bench/description/$desp_file"
#     done
# done
for i in {1..5}; do
    for desp_file in $(ls -p results/gvlmiqa_bench/description | grep -v '/$' | grep -v 'benchmark_2k.json' | grep -v 'preciseness_scores.*\.json'); do
        python eval/reason/description/gpt_eval/cal_gpt_score_preciseness_gvlmiqa.py --desp_file "results/gvlmiqa_bench/description/$desp_file"
    done
done
