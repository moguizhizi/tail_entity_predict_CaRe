#### Import all the supporting classes

import os
import time

from args import init_args
from model import ConvEParam
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

torch.backends.cudnn.enabled = False

from logger import config_logger

logger = config_logger('Model')

def get_triple_data(triple_file, is_use_cuda):
    fin = open(triple_file, "r", encoding="utf8").readlines()
    trip_list = []
    for trip in fin[0:]:
        temp = trip.strip().split()
        e1, r, e2 = int(temp[0]), int(temp[1]), int(temp[2])
        trip_list.append([e1, r, e2])

    tripls = np.array(trip_list)

    tripls = Variable(torch.from_numpy(tripls))
    if is_use_cuda:
        tripls = tripls.cuda()

    return tripls


if __name__ == '__main__':
    args = init_args()

    if args.use_glove:
        pad_entity_embedding = pad_glove_all_entity_embedding(args)
    else:
        pad_entity_embedding = pad_elmo_all_entity_embedding(args)

    for item in vars(args).items():
        logger.info('%s : %s', item[0], str(item[1]))

    rel2words = np.load(args.data_files['relation2word_path'])
    embed_matrix = np.load(args.data_files["word_glove_path"])
    entity_embedding = torch.Tensor(pad_entity_embedding)
    edge_index = torch.tensor(np.load(args.data_files['graph_edges_path']), dtype=torch.long)
    edge_type = torch.tensor(np.load(args.data_files['edges_type_path']), dtype=torch.long)
    edge_norm = edge_normalization(edge_type, edge_index, args.ent_total, args.rel_total)
    edge_norm = torch.tensor(edge_norm)

    model = ConvEParam(args, embed_matrix, rel2words)

    logger.info(model)

    if args.use_cuda:
        model.cuda()
        entity_embedding = entity_embedding.cuda()
        edge_index = edge_index.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()

    model_state_file = args.data_files['model_path']

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    train_triple = get_triple_data(args.data_files['train_trip_path'], args.use_cuda)
    valid_triple = get_triple_data(args.data_files['valid_trip_path'], args.use_cuda)
    test_triple = get_triple_data(args.data_files['test_trip_path'], args.use_cuda)

    train_id = np.arange(len(train_triple))

    best_MR = 20000
    best_MRR = 0
    best_epoch = 0
    count = 0
    label_graph = np.load(args.data_files['train_label_path'])
    for epoch in range(args.n_epochs):
        model.train()
        if count >= args.early_stop: break
        epoch_loss = 0
        permute = np.random.permutation(train_id)
        train_id = train_id[permute]
        n_batches = train_id.shape[0] // args.batch_size
        t1 = time.time()
        for i in range(n_batches):
            id_list = train_id[i * args.batch_size:(i + 1) * args.batch_size]
            samples, labels = get_next_batch(id_list, label_graph, args, train_triple)

            optimizer.zero_grad()
            loss = model.get_loss(samples, labels, entity_embedding, edge_index, edge_type, edge_norm)
            loss.backward()
            logger.info("batch {}/{} batches, batch_loss: {}".format(i, n_batches, (loss.data).cpu().numpy()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            epoch_loss += (loss.data).cpu().numpy()
        logger.info("epoch {}/{} total epochs, epoch_loss: {}".format(epoch + 1, args.n_epochs, epoch_loss / n_batches))

        if (epoch + 1) % args.eval_epoch == 0:
            model.eval()
            MR, MRR = evaluate(model, valid_triple, args, entity_embedding, label_graph, edge_index, edge_type, edge_norm)
            if MRR > best_MRR or MR < best_MR:
                count = 0
                if MRR > best_MRR: best_MRR = MRR
                if MR < best_MR: best_MR = MR
                best_epoch = epoch + 1
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            else:
                count += 1
            logger.info(
                "Best Valid MRR: {}, Best Valid MR: {}, Best Epoch(MR/MRR): {}".format(best_MRR, best_MR, best_epoch))
            scheduler.step(best_epoch)

    ### Get Embeddings
    logger.info("Test Set Evaluation ---")
    checkpoint = torch.load(model_state_file)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    _, _ = evaluate(model, test_triple, args, entity_embedding, label_graph, edge_index, edge_type, edge_norm)
