import torch

def qpu_linear(input, weight, bias):
    """
    input: (*, C_in * 4) -> (*, C_out * 4)
    """
    in_channels = input.shape[-1]
    in_channels = in_channels // 4
    out_channels = weight.shape[0]

    r, i, j, k = input.unsqueeze(-2).split(in_channels, dim=-1)

    r, i, j, k = quaternion_power_bias(r, i, j, k, weight, bias)
    r, i, j, k = QuaternionRemoveZeros.apply(r, i, j, k)
    r, i, j, k = quaternion_chained_prod(r, i, j, k, -1)
    # We can also use the custom autograd function which significantly decrease GPU memory usage (when QPU layers become deep), but is slower.
    # r, i, j, k = QuaternionChainedProdFunction.apply(r, i, j, k, -1)

    return torch.cat([r, i, j, k], dim=-1)


class QuaternionRemoveZeros(torch.autograd.Function):
    """Replace [0, 0, 0, 0] with [1, 0, 0, 0]
    """
    @staticmethod
    def forward(ctx,r,i,j,k):
        norm = r**2+ i**2+ j**2+ k**2
        index = norm == 0
        ctx.save_for_backward(index)
        r[index] = 1
        return r,i,j,k

    @staticmethod
    def backward(ctx,gr,gi,gj,gk):
        index, = ctx.saved_tensors
        gr[index] = 0
        gi[index] = 0
        gj[index] = 0
        gk[index] = 0
        return gr, gi, gj, gk


def quaternion_normalize(input, dim):
    """ Normalize quaternion
    """
    in_channels = input.shape[dim] // 4
    r, i, j, k = input.split(in_channels, dim)
    norm = torch.sqrt(r**2 + i**2 + j**2 + k**2 + 1e-12)
    r = r / norm
    i = i / norm
    j = j / norm
    k = k / norm
    return torch.cat([r, i, j, k], dim=dim)


def quaternion_power_bias(r, i, j, k, weight, bias):
    """
    r, i, j, k: (*, 1, C_in)
    weight: (C_out, C_in)
    bias: (C_out)
    return: [cos(w * (acos(r) + bias)), sin(w * (acos(r) + bias)) v / |v|]
    """
    # Compute new theta
    norm_v = torch.sqrt(i**2 + j**2 + k**2 + 1e-12)
    theta = torch.acos(torch.clamp(r, min=-1+1e-6, max=1-1e-6))
    if bias is not None:
        theta = theta + bias.unsqueeze(-1)
    theta = weight * theta
    
    mul = torch.sin(theta) / norm_v
    r = torch.cos(theta)
    i = i * mul
    j = j * mul
    k = k * mul
    return r, i, j, k


def quaternion_power(r, i, j, k, w):
    """
    r, i, j, k: (..., C_in, ...)
    w: (..., C_in, ...)
    return: [cos(w * acos(r)), sin(w * acos(r)) v / |v|]
    """    
    # Compute new theta
    norm_v = torch.sqrt(i**2 + j**2 + k**2 + 1e-12)
    theta = w * torch.acos(torch.clamp(r, min=-1+1e-6, max=1-1e-6))
    # Compute new quaternion
    r = torch.cos(theta)
    mul = torch.sin(theta) / norm_v
    i = i * mul
    j = j * mul
    k = k * mul
    return r, i, j, k


def quaternion_chained_prod_loop(r_input, i_input, j_input, k_input, dim=-1):
    """
    Chained quaternion product along a dimension (for loop)
    Hamilton product:
    a1 a2 - b1 b2 - c1 c2 - d1 d2 
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
    """
    seq_len = r_input.shape[dim]
    r_out, i_out, j_out, k_out = r_input.select(dim, 0), i_input.select(dim, 0), j_input.select(dim, 0), k_input.select(dim, 0)
    for i in range(1, seq_len):
        r_out, i_out, j_out, k_out = hamilton_product_chunk(r_out, i_out, j_out, k_out, r_input.select(dim, i), i_input.select(dim, i), j_input.select(dim, i), k_input.select(dim, i))

    return r_out, i_out, j_out, k_out


def quaternion_chained_prod(r_input, i_input, j_input, k_input, dim, last=None):
    """
    Chained quaternion product along a dimension (recursive)
    Hamilton product:
    a1 a2 - b1 b2 - c1 c2 - d1 d2 
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
    """
    channel = r_input.shape[dim]
    if channel == 1:
        return r_input.squeeze(dim), i_input.squeeze(dim), j_input.squeeze(dim), k_input.squeeze(dim)
    else:
        # Split into pair(0) and odd(1)
        r_out, i_out, j_out, k_out = r_input.unfold(dim, 2, 2), i_input.unfold(dim, 2, 2), j_input.unfold(dim, 2, 2), k_input.unfold(dim, 2, 2)
        r_pair, r_odd = r_out.select(-1, 0), r_out.select(-1, 1)
        i_pair, i_odd = i_out.select(-1, 0), i_out.select(-1, 1)
        j_pair, j_odd = j_out.select(-1, 0), j_out.select(-1, 1)
        k_pair, k_odd = k_out.select(-1, 0), k_out.select(-1, 1)
        # pair * odd
        r_out, i_out, j_out, k_out = hamilton_product_chunk(r_pair, i_pair, j_pair, k_pair, r_odd, i_odd, j_odd, k_odd)
        # Multiply last
        if channel % 2 == 1:
            last = (r_input.select(dim, -1), i_input.select(dim, -1), j_input.select(dim, -1), k_input.select(dim, -1))
        if r_out.shape[dim] % 2 == 1 and last is not None:
            r_out = torch.cat([r_out,last[0].unsqueeze(dim)],dim=dim)
            i_out = torch.cat([i_out,last[1].unsqueeze(dim)],dim=dim)
            j_out = torch.cat([j_out,last[2].unsqueeze(dim)],dim=dim)
            k_out = torch.cat([k_out,last[3].unsqueeze(dim)],dim=dim)
            last = None
        # Recursion
        r_out, i_out, j_out, k_out = quaternion_chained_prod(r_out, i_out, j_out, k_out, dim, last)
        return r_out, i_out, j_out, k_out


class QuaternionChainedProdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_r, input_i, input_j, input_k, dim=-1):
        """
        Chained quaternion product along a dimension (for loop)
        Hamilton product:
        a1 a2 - b1 b2 - c1 c2 - d1 d2 
        + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
        + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
        + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
        """
        input_r, input_i, input_j, input_k = input_r.clone(), input_i.clone(), input_j.clone(), input_k.clone()
        cumprod_r, cumprod_i, cumprod_j, cumprod_k = quaternion_cumprod_(input_r, input_i, input_j, input_k, dim)
        ctx.save_for_backward(cumprod_r, cumprod_i, cumprod_j, cumprod_k)
        ctx.dim = dim
        return cumprod_r.select(dim, -1), cumprod_i.select(dim, -1), cumprod_j.select(dim, -1), cumprod_k.select(dim, -1)

    @staticmethod
    def backward(ctx, grad_output_r, grad_output_i, grad_output_j, grad_output_k):
        cumprod_r, cumprod_i, cumprod_j, cumprod_k, = ctx.saved_tensors  # L, *
       
        # Compute cumprod of left and right seq for each input, grads are stored in cumprod on the fly to save memory
        grad_chain_r, grad_chain_i, grad_chain_j, grad_chain_k = quaternion_chained_prod_grad_cumprod(cumprod_r, cumprod_i, cumprod_j, cumprod_k, 
                                                                        grad_output_r, grad_output_i, grad_output_j, grad_output_k, dim=ctx.dim)
        
        return grad_chain_r, grad_chain_i, grad_chain_j, grad_chain_k, None


def hamilton_product_chunk(r1, i1, j1, k1, r2, i2, j2, k2):
    """
    Hamilton product
    a1 a2 - b1 b2 - c1 c2 - d1 d2 
    + ( a1 b2 + b1 a2 + c1 d2 − d1 c2 ) i
    + ( a1 c2 − b1 d2 + c1 a2 + d1 b2 ) j 
    + ( a1 d2 + b1 c2 − c1 b2 + d1 a2 ) k 
    """
    r_out, i_out, j_out, k_out = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2, \
                                 r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2, \
                                 r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2, \
                                 r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2
    return r_out, i_out, j_out, k_out


def quaternion_cumprod_(r, i, j, k, dim):
    """Cumpute quaternion cumpord (inplace)
    """
    seq_len = r.shape[dim]
    cumprod_r = r.split(1, dim)
    cumprod_i = i.split(1, dim)
    cumprod_j = j.split(1, dim)
    cumprod_k = k.split(1, dim)
    for n in range(1, seq_len):
        cr, ci, cj, ck = hamilton_product_chunk(cumprod_r[n - 1], cumprod_i[n - 1], cumprod_j[n - 1], cumprod_k[n - 1], 
                                                cumprod_r[n], cumprod_i[n], cumprod_j[n], cumprod_k[n])
        cumprod_r[n].copy_(cr)
        cumprod_i[n].copy_(ci)
        cumprod_j[n].copy_(cj)
        cumprod_k[n].copy_(ck)
    return r, i, j, k


def quaternion_chained_prod_grad_cumprod(cumprod_r, cumprod_i, cumprod_j, cumprod_k, grad_output_r, grad_output_i, grad_output_j, grad_output_k, dim):
    """Compute grad of quaternion chained prod from cumprod
    Args:
        cumprod_*: *, N, *
        grad_output_*: *, *
    """
    seq_len = cumprod_r.shape[dim]
    # Split shares the origin memory
    grad_output_r = grad_output_r.unsqueeze(dim)
    grad_output_i = grad_output_i.unsqueeze(dim)
    grad_output_j = grad_output_j.unsqueeze(dim)
    grad_output_k = grad_output_k.unsqueeze(dim)

    rl = torch.ones_like(cumprod_r)
    rl.narrow(dim, 1, seq_len - 1).copy_(cumprod_r.narrow(dim, 0, seq_len - 1))
    il = torch.zeros_like(cumprod_i)
    il.narrow(dim, 1, seq_len - 1).copy_(cumprod_i.narrow(dim, 0, seq_len - 1))
    jl = torch.zeros_like(cumprod_j)
    jl.narrow(dim, 1, seq_len - 1).copy_(cumprod_j.narrow(dim, 0, seq_len - 1))
    kl = torch.zeros_like(cumprod_k)
    kl.narrow(dim, 1, seq_len - 1).copy_(cumprod_k.narrow(dim, 0, seq_len - 1))

    rr, ir, jr, kr =  hamilton_product_chunk(cumprod_r, -cumprod_i, -cumprod_j, -cumprod_k, 
                                             cumprod_r.narrow(dim, seq_len - 1, 1), cumprod_i.narrow(dim, seq_len - 1, 1), 
                                             cumprod_j.narrow(dim, seq_len - 1, 1), cumprod_k.narrow(dim, seq_len - 1, 1))

    grad_r, grad_i, grad_j, grad_k = quaternion_chained_prod_grad(rl, il, jl, kl, rr, ir, jr, kr, 
                                                grad_output_r, grad_output_i, grad_output_j, grad_output_k)
    return grad_r, grad_i, grad_j, grad_k


def quaternion_chained_prod_grad(rl, il, jl, kl, rr, ir, jr, kr, grad_output_r, grad_output_i, grad_output_j, grad_output_k):
    grad_input_r = (   rl * rr - il * ir - jl * jr - kl * kr) * grad_output_r + \
                    (- ir * jl + il * jr + rr * kl + rl * kr) * grad_output_k + \
                    (  rr * jl + rl * jr + ir * kl - il * kr) * grad_output_j + \
                    (  rr * il + rl * ir - jr * kl + jl * kr) * grad_output_i

    grad_input_i = ( - rr * il - rl * ir - jr * kl + jl * kr) * grad_output_r + \
                    (- rr * jl + rl * jr - ir * kl - il * kr) * grad_output_k + \
                    (- ir * jl - il * jr + rr * kl - rl * kr) * grad_output_j + \
                    (  rl * rr - il * ir + jl * jr + kl * kr) * grad_output_i

    grad_input_j = ( - rr * jl - rl * jr + ir * kl - il * kr) * grad_output_r + \
                    (  rr * il - rl * ir - jr * kl - jl * kr) * grad_output_k + \
                    (  rl * rr + il * ir - jl * jr + kl * kr) * grad_output_j + \
                    (- ir * jl - il * jr - rr * kl + rl * kr) * grad_output_i

    grad_input_k = ( - ir * jl + il * jr - rr * kl - rl * kr) * grad_output_r + \
                    (  rl * rr + il * ir + jl * jr - kl * kr) * grad_output_k + \
                    (- rr * il + rl * ir - jr * kl - jl * kr) * grad_output_j + \
                    (  rr * jl - rl * jr - ir * kl - il * kr) * grad_output_i
    
    return grad_input_r, grad_input_i, grad_input_j, grad_input_k


# Tests
if __name__ == '__main__':
    print('>>>Test quaternion_chained_prod')
    from utils.quaternion import q_mul
    input = torch.randn(2, 4, 21)
    r_input, i_input, j_input, k_input = input[:, 0, :], input[:, 1, :], input[:, 2, :], input[:, 3, :]
    input_np = input.detach().numpy()
    gt = [input_np[0, :, 0], input_np[1, :, 0]]
    for batch in range(2):
        for i in range(1, 21):
            gt[batch] = q_mul(gt[batch], input_np[batch, :, i])
    print('gt', gt)
    r_out, i_out, j_out, k_out = quaternion_chained_prod(r_input, i_input, j_input, k_input, -1)
    print('quaternion_chained_prod', r_out, i_out, j_out, k_out)

    print('>>>Test QuaternionChainedProdFunction')
    from utils.quaternion import q_mul
    B = 2
    N = 21
    input = torch.randn(B, 4, N)
    norm = torch.norm(input, dim=1, keepdim=True)
    input = input / norm
    input.requires_grad=True
    r_input, i_input, j_input, k_input = input[:, 0, :], input[:, 1, :], input[:, 2, :], input[:, 3, :]
    input_np = input.detach().numpy()
    gt = [x for x in input_np[:, :, 0]]
    # gt = [input_np[0, :, 0], input_np[1, :, 0]]
    for batch in range(B):
        for i in range(1, N):
            gt[batch] = q_mul(gt[batch], input_np[batch, :, i])
    print('gt', gt)
    print(input)
    r_out, i_out, j_out, k_out = QuaternionChainedProdFunction.apply(r_input, i_input, j_input, k_input, -1)
    print('QuaternionChainedProdFunction', r_out.detach().numpy(), i_out.detach().numpy(), j_out.detach().numpy(), k_out.detach().numpy())
    
    # Compare Grads
    print('>>> Test QuaternionChainedProdFunction backward')
    loss = torch.sum(r_out + i_out + j_out + k_out)
    print('loss Function', loss)
    loss.backward()
    grad1 = input.grad.clone()
    input.grad = None
    r_input, i_input, j_input, k_input = input[:, 0, :], input[:, 1, :], input[:, 2, :], input[:, 3, :]
    r_out, i_out, j_out, k_out = quaternion_chained_prod_loop(r_input, i_input, j_input, k_input, -1)

    loss = torch.sum(r_out + i_out + j_out + k_out)
    print('loss autograd', loss)
    loss.backward()
    grad2 = input.grad.clone()
    print('grad Function', grad1)
    print('grad autograd:', grad2)
    print('grad diff ', (grad1 - grad2).abs().max())
