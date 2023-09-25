import logging
import config
from utils import utils
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import os
from dataloader.data import get_dataset
from torch.utils.data import DataLoader
from approaches.eval import Appr

logger = logging.getLogger(__name__)

args = config.parse_args()
args = utils.prepare_sequence_eval(args)

accelerator = Accelerator()

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
accelerator.wait_for_everyone()

dataset = get_dataset(args)
model = utils.lookfor_model(args)

test_loaders = []
train_loaders = []
for eval_t in range(args.ntasks):
    test_dataset = dataset[eval_t]['test']
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loaders.append(test_dataloader)
    train_dataset = dataset[eval_t]['train']
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    train_loaders.append(train_dataloader)

replay_loader = DataLoader(dataset[args.task+1]['replay'], batch_size=int((args.replay_buffer_size // (args.class_num * (args.task + 1))) * args.class_num), shuffle=False, num_workers=8)

appr = Appr(args)
appr.eval(model, train_loaders, test_loaders, replay_loader, accelerator)