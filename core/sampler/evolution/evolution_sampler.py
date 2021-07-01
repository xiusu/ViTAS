import numpy as np
from core.sampler.evolution import nsganet as engine

from pymop.problem import Problem
from pymoo.optimize import minimize
from core.sampler.base_sampler import BaseSampler
from core.utils.flops import count_flops
from core.utils.measure import measure_model
import torch.distributed as dist
import torch
import random
import copy


class EvolutionSampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

        if not hasattr(self, 'heads_share'):
            self.heads_share = False

        assert self.heads_share == True, f'We need share heads for Block opeartion'

        if not hasattr(self, 'GPU_search'):
            self.GPU_search = False
        
        self.init_evolution()

        if self.GPU_search:
            self.prop_search_GPU()
        else:
            self.prop_search()
        torch.distributed.barrier()
        self.num = 0        # print the count of current iter
        assert self.side_scheme in ['BCNet', 'AutoSlim']

    def prop_search(self):
        self.id_prop = 0.5
        if self.rank == 0:
            print(f'start prop_search: {self.id_prop}')

        while(True):
            FLOPs_count = 0
            for i in range(self.ps_num):
                subnet = self.generate_subnet()
                FLOPs_count += count_flops(self.model, subnet_m = subnet[:self.len_block], subnet_c = self.covert_channels(subnet[self.len_block:]), heads_share = self.heads_share)
            
            FLOPs_count = FLOPs_count / self.ps_num
            if FLOPs_count > self.flops_constraint:
                self.id_prop = self.id_prop + 0.01
                if self.rank == 0:
                    print(f'Too large, id_prop: {self.id_prop}, FLOPs_count: {FLOPs_count}, flops_constraint: {self.flops_constraint}')
            elif FLOPs_count < self.flops_min:
                self.id_prop = self.id_prop - 0.01
                if self.rank == 0:
                    print(f'Too small, id_prop: {self.id_prop}, FLOPs_count: {FLOPs_count}, flops_constraint: {self.flops_min}')
            else:
                print("all_right, id_prop is {}".format(self.id_prop))
                break
            if self.id_prop >= 1 or self.id_prop <= 0:
                raise RuntimeError(f'stop, id_prop: {self.id_prop}')
    
    def prop_search_GPU(self):
        self.id_prop = 0.5
        if self.rank == 0:
            print(f'start prop_search_GPU: {self.id_prop}')

        finished_GPU = torch.Tensor([0]).cuda()
        if self.rank == 0:
            while(True):
                Pics_count = 0
                time_cost = []

                for i in range(self.ps_num):
                    subnet = self.generate_subnet()
                    subnet_m = subnet[:self.len_block]
                    subnet_c = self.covert_channels(subnet[self.len_block:])
                    subnet_m.extend(subnet_c)
                    subnet = subnet_m
                    time_cost.append(measure_model(self.model, subnet = subnet))

                Pics_count = sum(time_cost[(self.ps_num//2-3): (self.ps_num//2+3)]) / 6
                if Pics_count > self.GPUs_constraint:
                    self.id_prop = self.id_prop - 0.01
                    if self.rank == 0:
                        print(f'Too large, id_prop: {self.id_prop}, Pics_count: {Pics_count}, GPUs_constraint: {self.GPUs_constraint}')
                    finished_GPU = torch.Tensor([0]).cuda()
                    dist.broadcast(finished_GPU,0)
                elif Pics_count < self.GPUs_min:
                    self.id_prop = self.id_prop + 0.01
                    if self.rank == 0:
                        print(f'Too small, id_prop: {self.id_prop}, Pics_count: {Pics_count}, GPUs_min: {self.GPUs_min}')
                    finished_GPU = torch.Tensor([0]).cuda()
                    dist.broadcast(finished_GPU,0)
                else:
                    print("all_right, id_prop is {}".format(self.id_prop))
                    finished_GPU = torch.Tensor([1]).cuda()
                    dist.broadcast(finished_GPU,0)
                    break
                if self.id_prop >= 1 or self.id_prop <= 0:
                    raise RuntimeError(f'stop, id_prop: {self.id_prop}')
        else:
            while(True):
                finished_GPU = torch.Tensor([0]).cuda()
                for i in range(self.ps_num):
                    subnet = self.generate_subnet()
                    subnet_m = subnet[:self.len_block]
                    subnet_c = self.covert_channels(subnet[self.len_block:])
                    subnet_m.extend(subnet_c)
                    subnet = subnet_m
                    measure_model(self.model, subnet = subnet)
                dist.broadcast(finished_GPU,0)
                if finished_GPU[0] == 1:
                    break




    def covert_channels(self, subnet_c):
        subnet_percent = []
        for i in subnet_c:
            subnet_percent.append(self.channel_percent[i])
        return subnet_percent




    def generate_subnet(self):
        assert self.id_prop is not None, "id_prop should be a small number"
        subnet_m = []
        subnet_c = []


        for name, block in self.model.module.net.named_children():
            # for subnet_m
            if 'id' in name or 'Block' in name:
                # id op
                if random.random() < self.id_prop:
                    if self.heads_share:
                        subnet_m.append(len(block) - 1 + 3) # 3 different heads
                    else:
                        subnet_m.append(len(block) - 1)
                # norm op
                else:
                    if self.heads_share:
                        subnet_m.append(random.randint(0,len(block) - 2 + 3))
                    else:
                        subnet_m.append(random.randint(0, max(len(block)-2, 0)))
            elif 'Patch_init' in name:
                subnet_m.append(random.randint(0, len(block)-1))
            else:
                subnet_m.append(0)
            
            # for subnet_c
            if 'Patch_init' in name:
                subnet_c.append(random.randint(0, len(self.channel_percent)-1))
            if 'Block' in name or 'id' in name:
                subnet_c.append(random.randint(0, len(self.channel_percent)-1))
                subnet_c.append(random.randint(0, len(self.channel_percent)-1))

        subnet_m = torch.IntTensor(subnet_m).cuda()
        subnet_c = torch.IntTensor(subnet_c).cuda()
        dist.broadcast(subnet_m,0)   
        dist.broadcast(subnet_c,0)             
        subnet_m = subnet_m.tolist()
        subnet_c = subnet_c.tolist()

        subnet_m.extend(subnet_c)
        return subnet_m

    def init_evolution(self):
        self.name_list = []
        self.block_list = []
        
        self.block_len = []
        self.channels_len = []
        
        for name, block in self.model.module.net.named_children():
            self.name_list.append(name)
            self.block_list.append(block)

            if 'id' in name or 'Block' in name:
                if self.heads_share:
                    self.block_len.append(len(block) - 1 + 3)
                else:
                    self.block_len.append(len(block) - 1)
            else:
                self.block_len.append(len(block) - 1)

            if 'Patch_init' in name:
                self.channels_len.append(len(self.channel_percent)-1)
            elif 'Block' in name or 'id' in name:
                self.channels_len.append(len(self.channel_percent)-1)
                self.channels_len.append(len(self.channel_percent)-1)
            
        all_list = copy.deepcopy(self.block_len)
        all_list.extend(self.channels_len)
        self.n_var = len(all_list)
        self.lb = np.zeros(self.n_var)
        self.ub = np.array(all_list, dtype = float)

        self.len_block = len(self.name_list)




    def init_population(self):
        initial_pop = []
        if self.rank == 0:
            print("start init_population")
        while len(initial_pop) < self.pop_size:
            subnet = self.generate_subnet()
            flops = count_flops(self.model, subnet_m = subnet[:self.len_block], subnet_c = self.covert_channels(subnet[self.len_block:]), heads_share = self.heads_share)
            if flops <= self.flops_constraint and flops >= self.flops_min:
                if self.rank == 0:
                    print('adopt subnet: {}, FLOPs: {}'.format(subnet, flops))
                initial_pop.append(subnet)
            else:
                if self.rank == 0:
                    print('not adopt subnet: {}, FLOPs: {}'.format(subnet, flops))
        initial_pop = np.array(initial_pop, dtype=np.int)
        if self.rank == 0:
            print("init_population done")
        return initial_pop



    def init_population_GPU(self):
        initial_pop = []
        if self.rank == 0:
            print("start init_population_GPU")
        while len(initial_pop) < self.pop_size:
            subnet = self.generate_subnet()

            subnet_m = subnet[:self.len_block]
            subnet_c = self.covert_channels(subnet[self.len_block:])
            subnet_m.extend(subnet_c)

            Pics_count = measure_model(self.model, subnet = subnet_m)
            Pics_count = int(Pics_count)
            Pics_count = torch.IntTensor([Pics_count]).cuda()
            dist.broadcast(Pics_count,0)   
            Pics_count = Pics_count[0].item()
            if Pics_count <= self.GPUs_constraint and Pics_count >= self.GPUs_min:
                if self.rank == 0:
                    print('adopt subnet: {}, Pics_count: {}'.format(subnet, Pics_count))
                initial_pop.append(subnet)
            else:
                if self.rank == 0:
                    print('not adopt subnet: {}, Pics_count: {}'.format(subnet, Pics_count))
        initial_pop = np.array(initial_pop, dtype=np.int)
        if self.rank == 0:
            print("init_population done")
        return initial_pop

    def eval_subnet_host(self, subnet):
        finished = torch.Tensor([0]).cuda()
        dist.broadcast(finished, 0)  # not finished

        # broadcast subnet, then run together
        dist.broadcast(torch.Tensor(subnet).cuda(), 0)
        score = self.eval_subnet(subnet)
        return score

    def sample(self, sampling=None):
        '''
        @sampling: initial population: 2d numpy array, dtype:np.int
        '''
        if getattr(self, 'init_pop', True):
            if self.GPU_search:
                sampling = self.init_population_GPU()
            else:
                sampling = self.init_population()
        else:
            raise RuntimeError("need a init_population")

        if self.rank ==0:
            print("initial_population Done")

        subnet_eval_dict = {}
        if self.rank == 0:
            n_offspring = None #
            nas_problem = NAS(n_var=self.n_var, n_obj=1, n_constr=0, lb=self.lb, ub=self.ub,
                              eval_func=lambda subnet: self.eval_subnet_host(subnet),
                              result_dict=subnet_eval_dict)

            # configure the nsga-net method
            if sampling is not None:
                method = engine.nsganet(pop_size=self.pop_size,
                                        n_offsprings=n_offspring,
                                        eliminate_duplicates=True,
                                        sampling=sampling)
            else:
                method = engine.nsganet(pop_size=self.pop_size,
                                        n_offsprings=n_offspring,
                                        eliminate_duplicates=True)

            res = minimize(nas_problem,
                           method,
                           callback=lambda algorithm: self.generation_callback(algorithm),
                           termination=('n_gen', self.n_gens))
        else:
            # slaver: wait for signal
            while True:
                finished = torch.Tensor([0]).cuda()
                dist.broadcast(finished, 0)
                if finished[0] == 1:
                    break

                # get subnet
                subnet = torch.zeros([self.n_var]).cuda()
                dist.broadcast(subnet, 0)
                subnet = [int(x) for x in subnet.tolist()]
                self.eval_subnet(subnet)

        # finished
        if self.rank == 0:
            finished = torch.Tensor([1]).cuda()
            dist.broadcast(finished, 0)

        subnet_topk = []
        if self.rank == 0:
            sorted_subnet = sorted(subnet_eval_dict.items(), key=lambda i: i[1], reverse=True)
            sorted_subnet_key = [x[0] for x in sorted_subnet]
            subnet_topk = sorted_subnet_key[:self.sample_num]
            self.subnet_top1 = sorted_subnet_key[:1]
            if self.rank == 0:
                print('== search result ==')
                print(sorted_subnet)
                print('== best subnet ==')
                print(subnet_topk)
                print('== id_prop ==')
                print(self.id_prop)
            self.subnet_topk = subnet_topk
        
        raise RuntimeError('sampling over, please check the answer')

    def generation_callback(self, algorithm):
        gen = algorithm.n_gen
        pop_var = algorithm.pop.get("X")
        pop_obj = algorithm.pop.get("F")
        print(f'==Finished generation: {gen}')


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None, eval_func=None, result_dict=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int, )
        self.xl = lb
        self.xu = ub
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.eval_func = eval_func
        self.result_dict = result_dict

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1

            # all objectives assume to be MINIMIZED !!!!!
            if self.result_dict.get(str(x[i])) is not None:
                acc = self.result_dict[str(x[i])]
            else:
                acc = self.eval_func(x[i])
                self.result_dict[str(x[i])] = acc

            print('==evaluation subnet:{} prec@1:{}'.format(str(x[i]), acc))

            objs[i, 0] = 100 - acc  # performance['valid_acc']
            # objs[i, 1] = 10  # performance['flops']

            self._n_evaluated += 1
        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")
    print(gen)
    # print(gen, pop_var, pop_obj)
    # report generation info to files

def main():
    # hyper parameters
    pop_size = 50
    n_gens = 20
    n_offspring = 40

    # setup NAS search problem
    n_var = 20
    lb = np.zeros(n_var)  # left index of each block
    ub = np.zeros(n_var) + 4  # right index of each block

    nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub)

    # configure the nsga-net method
    method = engine.nsganet(pop_size=pop_size,
                            n_offsprings=n_offspring,
                            eliminate_duplicates=True)

    res = minimize(nas_problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', n_gens))
    print(dir(res))
    print(len(res.pop))
    for pop in res.pop[:10]:
        print(pop.F, pop.X)
    return res


if __name__ == "__main__":
    main()
