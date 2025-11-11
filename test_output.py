# test_output.py
import pyarrow.parquet as pq

# PATH = "outputs/submissions/humob_submission_parquet.parquet"
PATH = "outputs/eval/pred_gt_pairs_val.parquet"

pf = pq.ParquetFile(PATH)
print("row_groups:", pf.metadata.num_row_groups)
print("rows:", pf.metadata.num_rows)      # conta total, sem carregar dados
print("schema:", pf.schema_arrow)
