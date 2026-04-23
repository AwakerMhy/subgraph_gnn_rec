"""预处理所有数据集。"""
import sys
sys.path.insert(0, ".")

from src.dataset.real.bitcoin_otc    import BitcoinOTCDataset
from src.dataset.real.email_eu       import EmailEUDataset
from src.dataset.real.college_msg    import CollegeMsgDataset
from src.dataset.real.bitcoin_alpha  import BitcoinAlphaDataset
from src.dataset.real.dnc_email      import DNCEmailDataset
from src.dataset.real.sx_mathoverflow import SXMathOverflowDataset
from src.dataset.real.sx_askubuntu   import SXAskUbuntuDataset
from src.dataset.real.sx_superuser   import SXSuperUserDataset
from src.dataset.real.gowalla        import GowallaDataset
from src.dataset.real.epinions       import EpinionsDataset

datasets = [
    ("Bitcoin-OTC",      BitcoinOTCDataset),
    ("Email-EU",         EmailEUDataset),
    ("CollegeMsg",       CollegeMsgDataset),
    ("Bitcoin-Alpha",    BitcoinAlphaDataset),
    ("DNC-Email",        DNCEmailDataset),
    ("SX-MathOverflow",  SXMathOverflowDataset),
    ("SX-AskUbuntu",     SXAskUbuntuDataset),
    ("SX-SuperUser",     SXSuperUserDataset),
    ("Gowalla",          GowallaDataset),
    ("Epinions",         EpinionsDataset),
]

for label, cls in datasets:
    print(f"\n=== {label} ===")
    try:
        ds = cls()
        ds.processed_dir.mkdir(parents=True, exist_ok=True)
        ds.process()
    except AssertionError as e:
        print(f"  跳过：{e}")
    except Exception as e:
        print(f"  错误：{e}")

print("\n所有数据集预处理完成。")
