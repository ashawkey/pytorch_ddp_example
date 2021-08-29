# pytorch DDP example

### requirements
pytorch >= 1.8

### features
* mixed precision training (native `amp`)
* DDP training (use `mp.spawn` to call)
* DDP inference (`all_gather` statistics from all threads)
