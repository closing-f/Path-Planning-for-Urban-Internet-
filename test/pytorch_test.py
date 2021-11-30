import sys 
sys.path.append("..") 
from utils.arguments import parse_args
config =parse_args()
print(config.device)