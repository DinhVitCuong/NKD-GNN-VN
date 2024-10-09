import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
# and save them in some local working folder
py_vncorenlp.download_model(save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')

# Load VnCoreNLP from the local working folder that contains both `VnCoreNLP-1.2.jar` and `models` 
model = py_vncorenlp.VnCoreNLP(save_dir=r'D:\Study\DATN\model\NKD-GNN-test\VnCoreNLP')
# Equivalent to: model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir='/absolute/path/to/vncorenlp')

# Annotate a raw text
model.print_out(model.annotate_text("Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."))