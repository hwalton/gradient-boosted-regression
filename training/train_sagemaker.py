import sagemaker
from sagemaker.sklearn import SKLearn
import os

class Cfg:
    sess: str = sagemaker.Session()
    role: str = os.environ.get("ROLE", f"arn:aws:iam::{os.environ.get['USER']}:role/rot")

def main():
    sk = SKLearn(
        entry_point="training/sagemaker_entry.py",
        source_dir=".",                # repo root (uploads code)
        role=Cfg.role,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=Cfg.sess,
    )

    s3_input = f"s3://{os.environ['BUCKET']}/processed/"
    sk.fit({"training": s3_input})

if __name__ == "__main__":
    main()