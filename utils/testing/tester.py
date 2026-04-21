import torch

class Tester(object):

    def test(self, model, dataloader, args, accelerator=None):
        model.eval()
        step = 0

        y_out_list = []
        idxs_list = []
        with torch.no_grad():
            for graph in dataloader:

                idx = torch.atleast_2d(torch.tensor(graph.idxs, device=accelerator.device))
                idxs_list.append(idx)
                
                y_out = model(graph)
                y_out_list.append(y_out)

                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        y_out = torch.stack(y_out_list).squeeze()
        idxs = torch.stack(idxs_list).squeeze()

        return y_out, idxs