import torch
import torch.nn as nn
import torch.nn.functional as F

def euclid(a, b):
    return torch.norm(a - b, p=2, dim=1)

class TetraLoss(nn.Module):
    def __init__(self, margin=3.0):
        super().__init__()
        self.margin = margin

    def forward(self, xa, xp, xn, xm, reduction="mean"):
        """so

        :param xa: anchor embeddings (L2-normalized)
        :param xp: positive embeddings
        :param xn: negative embeddings
        :param xm: morph embeddings
        :param reduction: "mean" (default) or "none"
        """
        dap = euclid(xa, xp)
        dan = euclid(xa, xn)
        dam = euclid(xa, xm)

        hardest = torch.minimum(dan, dam)
        loss = F.relu(dap + self.margin - hardest)

        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")


class TetraLossExt(nn.Module):
    """TetraLoss + morph-attraction term.

    loss = clamp(d_ap + margin - min(d_an, d_am)) + lam * d(x_m, sg(x_n))

    The stop-gradient on x_n is mandatory: without it the attraction term
    also pulls the negative toward the morph, destabilising bona fide separation.
    lam=0.0 reproduces plain TetraLoss exactly.
    Recommended starting range: lam in {0.01, 0.1, 0.5}.
    """
    def __init__(self, margin=3.0, lam=0.1):
        super().__init__()
        self.margin = margin
        self.lam = lam

    def forward(self, xa, xp, xn, xm, reduction="mean"):
        dap = euclid(xa, xp)
        dan = euclid(xa, xn)
        dam = euclid(xa, xm)

        hardest = torch.minimum(dan, dam)
        loss_tetra = F.relu(dap + self.margin - hardest)
        loss_attract = euclid(xm, xn.detach())

        loss = loss_tetra + self.lam * loss_attract

        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

class TetraLossDirected(nn.Module):
    """DirectedTetraLoss: TetraLoss + impostor-directed attraction term (hinged).

    loss = relu(d_ap + margin - min(d_an, d_am)) + lam * relu(d(x_m, sg(x_n)) - d(x_a, x_m))

    The hinge on the attraction term (relu(d_mn - d_am)) is self-limiting:
    it only fires when the morph is farther from the negative than from the anchor,
    i.e. when the morph is still on the wrong side geometrically. Once d(m,n) < d(a,m)
    the term goes to zero and TetraLoss reclaims full optimisation capacity.

    The stop-gradient on x_n is mandatory: without it the attraction term also pulls
    the negative toward the morph, destabilising bona fide separation. Only x_m
    receives gradient from the attraction term.

    lam=0.0 reproduces plain TetraLoss exactly.
    Recommended starting range: lam in {0.1, 0.5, 1.0}.
    Recommended margin: 0.5–1.5 (m=3.0 makes zero loss geometrically infeasible
    on the unit hypersphere since max Euclidean distance = 2.0).

    Diagnostic: log d_mn alongside d_ap, d_an, d_am to verify the hinge is active.
    If median(d_mn) < median(d_am) the hinge fires rarely — fall back to raw attraction.
    """

    def __init__(self, margin=1.0, lam=0.5):
        super().__init__()
        self.margin = margin
        self.lam = lam

    def forward(self, xa, xp, xn, xm, reduction="mean"):
        dap = euclid(xa, xp)
        dan = euclid(xa, xn)
        dam = euclid(xa, xm)
        dmn = euclid(xm, xn.detach())   # stop-gradient on x_n

        hardest = torch.minimum(dan, dam)
        loss_tetra = F.relu(dap + self.margin - hardest)
        loss_attract = F.relu(dmn - dam)  # hinge: only fires when d(m,n) > d(a,m)

        loss = loss_tetra + self.lam * loss_attract

        if reduction == "none":
            return loss, loss_tetra, loss_attract
        if reduction == "mean":
            return loss.mean(), loss_tetra.mean(), loss_attract.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")


class TetraLossWorstCase(nn.Module):
    """TetraLoss + worst-case morph embedding term.

    L_total = L_tetra + lam * L_wc

    L_tetra = relu( d(a,p) + margin - min(d(a,n), d(a,m)) )
    L_wc    = relu( d(m, y*) + margin2 - d(a,m) )

    y* = normalize(y_a + y_b) — the worst-case embedding: the normalized midpoint
    of both contributor anchor embeddings, equidistant (in angular terms) from
    both contributors. y* is detached — only the morph receives gradient from L_wc.

    The hinge fires when d(m,y*) > d(a,m) - margin2, i.e. the morph is not yet
    close enough to y*. It switches off once the morph is sufficiently near y*,
    making L_wc self-limiting and non-competing with L_tetra.

    The attraction term from DirectedTetraLoss is omitted entirely.

    Args:
        margin:  TetraLoss margin (default 1.5 for unit-sphere runs)
        margin2: worst-case hinge margin (recommended start: 0.2)
        lam:     weight of worst-case term (recommended start: 0.5)
    """

    def __init__(self, margin=1.5, margin2=0.2, lam=0.5):
        super().__init__()
        self.margin = margin
        self.margin2 = margin2
        self.lam = lam

    def forward(self, za, zp, zn, zm, zb, reduction="mean"):
        """
        :param za: anchor embeddings of contributor A (L2-normalized)
        :param zp: positive embeddings
        :param zn: negative embeddings
        :param zm: morph embeddings
        :param zb: anchor embeddings of contributor B (L2-normalized)
        :param reduction: "mean" or "none"
        """
        dap = euclid(za, zp)
        dan = euclid(za, zn)
        dam = euclid(za, zm)

        hardest = torch.minimum(dan, dam)
        loss_tetra = F.relu(dap + self.margin - hardest)

        # Worst-case embedding: normalized midpoint of both contributor anchors.
        # Full stop-gradient — only zm receives gradient from loss_wc.
        y_star = F.normalize(za.detach() + zb.detach(), p=2, dim=1)

        d_m_ystar = euclid(zm, y_star)
        loss_wc = F.relu(d_m_ystar + self.margin2 - dam)

        loss = loss_tetra + self.lam * loss_wc

        if reduction == "none":
            return loss, loss_tetra, loss_wc
        if reduction == "mean":
            return loss.mean(), loss_tetra.mean(), loss_wc.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")


class TetraLossBalanced(nn.Module):
    """TetraLoss with 50:50 random hardest example selection.

    Standard TetraLoss always picks min(d(a,n), d(a,m)) — in practice
    the morph always wins for similarity-driven pairs, so the negative
    receives almost no gradient. This variant randomly selects either
    the morph or the negative with equal probability per sample per step,
    ensuring ~50% of gradient steps push negatives away from the anchor.

    Args:
        margin: TetraLoss margin (recommended: 1.5)
    """

    def __init__(self, margin=1.5):
        super().__init__()
        self.margin = margin

    def forward(self, za, zp, zn, zm, reduction="mean"):
        dap = euclid(za, zp)
        dan = euclid(za, zn)
        dam = euclid(za, zm)

        # One independent random draw per sample — fresh each forward call
        mask    = torch.rand(dan.shape[0], device=dan.device) > 0.5
        hardest = torch.where(mask, dan, dam)

        loss = F.relu(dap + self.margin - hardest)

        if reduction == "mean":
            return loss.mean()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {reduction}")


class TetraLossWorstCaseBalanced(nn.Module):
    """TetraLoss + WorstCase hinge with 50:50 balanced hardest example selection.

    Combines two fixes:
      1. Balanced selection: negative receives gradient in ~50% of steps
         instead of almost never (morph always won the min() in standard TetraLoss).
      2. WC term: directional morph displacement toward y* (identical to TetraLossWorstCase).

    Exactly one line differs from TetraLossWorstCase: `torch.minimum` → random mask.

    Args:
        margin:  TetraLoss margin m1 (recommended: 1.5)
        margin2: WC hinge margin m2 (recommended: 0.2)
        lam:     weight of WC term (recommended: 0.1)
    """

    def __init__(self, margin=1.5, margin2=0.2, lam=0.1):
        super().__init__()
        self.margin  = margin
        self.margin2 = margin2
        self.lam     = lam

    def forward(self, za, zp, zn, zm, zb, reduction="mean"):
        """
        :param za: anchor embeddings — contributor A (L2-normalized)
        :param zp: positive embeddings (L2-normalized)
        :param zn: negative embeddings (L2-normalized)
        :param zm: morph embeddings (L2-normalized)
        :param zb: contributor B anchor embeddings (L2-normalized)
        :param reduction: "mean" or "none"
        """
        dap = euclid(za, zp)
        dan = euclid(za, zn)
        dam = euclid(za, zm)

        # 50:50 balanced hardest example selection — fresh draw each forward call
        mask    = torch.rand(dan.shape[0], device=dan.device) > 0.5
        hardest = torch.where(mask, dan, dam)

        loss_tetra = F.relu(dap + self.margin - hardest)

        # WC component — identical to TetraLossWorstCase
        # Full stop-gradient: only zm receives gradient from loss_wc
        y_star    = F.normalize(za.detach() + zb.detach(), p=2, dim=1)
        d_m_ystar = euclid(zm, y_star)
        loss_wc   = F.relu(d_m_ystar + self.margin2 - dam)

        loss = loss_tetra + self.lam * loss_wc

        if reduction == "none":
            return loss, loss_tetra, loss_wc
        if reduction == "mean":
            return loss.mean(), loss_tetra.mean(), loss_wc.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")


class TripletWorstCase(nn.Module):
    """TripletLoss + worst-case morph embedding term.

    Decouples bona fide geometry from morph displacement:
      L_triplet handles genuine/impostor separation exclusively.
      L_wc      handles morph displacement exclusively.

    L_total   = L_triplet + lam * L_wc
    L_triplet = relu( d(a,p) + margin  - d(a,n) )
    L_wc      = relu( d(m,y*) + margin2 - d(a,m) )

    y* = normalize(za.detach() + zb.detach()) — worst-case embedding,
    equidistant (in angular terms) from both contributors.
    Only zm receives gradient from L_wc; za and zb are fully stopped.

    Args:
        margin:  triplet margin m1 (recommended start: 0.2)
        margin2: worst-case hinge margin m2 (recommended start: 0.2)
        lam:     weight of worst-case term (recommended start: 0.5)
    """

    def __init__(self, margin=0.2, margin2=0.2, lam=0.5):
        super().__init__()
        self.margin  = margin
        self.margin2 = margin2
        self.lam     = lam

    def forward(self, za, zp, zn, zm, zb, reduction="mean"):
        """
        :param za: anchor embeddings — contributor A (L2-normalized)
        :param zp: positive embeddings (L2-normalized)
        :param zn: negative embeddings (L2-normalized)
        :param zm: morph embeddings (L2-normalized)
        :param zb: contributor B anchor embeddings (L2-normalized)
        :param reduction: "mean" or "none"
        """
        dap = euclid(za, zp)
        dan = euclid(za, zn)
        dam = euclid(za, zm)

        loss_triplet = F.relu(dap + self.margin - dan)

        # Worst-case embedding: normalized midpoint of both contributor anchors.
        # Full stop-gradient — only zm receives gradient from loss_wc.
        y_star    = F.normalize(za.detach() + zb.detach(), p=2, dim=1)
        d_m_ystar = euclid(zm, y_star)
        loss_wc   = F.relu(d_m_ystar + self.margin2 - dam)

        loss = loss_triplet + self.lam * loss_wc

        if reduction == "none":
            return loss, loss_triplet, loss_wc
        if reduction == "mean":
            return loss.mean(), loss_triplet.mean(), loss_wc.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")


class TripletRepulsionLoss(nn.Module):
    """Triplet loss with symmetric morph repulsion from both contributors.

    L = relu( d(a,p) + margin       - d(a,n) )          # Term 1: standard triplet
      + lam * relu( d(a,p) + margin_repel - d(a,m) )    # Term 2: morph repelled from A
      + lam * relu( d(a,p) + margin_repel - d(b,m) )    # Term 3: morph repelled from B

    d(a,p) as reference scale makes morph repulsion adaptive to bona fide geometry.
    No stop-gradient — all five embeddings carry gradient.
    No y* / WC attraction term.

    Args:
        margin:       triplet hinge margin m1 (recommended: 1.0)
        margin_repel: morph repulsion hinge margin m2 (recommended: 0.3)
        lam:          weight on the two repulsion terms (recommended start: 0.5)
    """

    def __init__(self, margin: float = 1.0, margin_repel: float = 0.3, lam: float = 0.5):
        super().__init__()
        self.margin = margin
        self.margin_repel = margin_repel
        self.lam = lam

    def forward(self, za, zp, zn, zm, zb, reduction="mean"):
        """
        :param za: contributor A anchor embeddings (L2-normalized)
        :param zp: positive embeddings (different image of A, L2-normalized)
        :param zn: random non-contributor embeddings (L2-normalized)
        :param zm: morph embeddings (L2-normalized)
        :param zb: contributor B anchor embeddings (L2-normalized)
        :param reduction: "mean" or "none"
        """
        dap = euclid(za, zp)
        dan = euclid(za, zn)
        dam = euclid(za, zm)
        dbm = euclid(zb, zm)

        loss_triplet = F.relu(dap + self.margin - dan)
        loss_repel_a = F.relu(dap + self.margin_repel - dam)
        loss_repel_b = F.relu(dap + self.margin_repel - dbm)
        loss_repel   = loss_repel_a + loss_repel_b

        loss = loss_triplet + self.lam * loss_repel

        if reduction == "none":
            return loss, loss_triplet, loss_repel
        if reduction == "mean":
            return loss.mean(), loss_triplet.mean(), loss_repel.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")


class DirectedRepulsionLoss(nn.Module):
    """Combined TripletRepulsion + Directed loss.

    L = relu( d(a,p) + m1       - d(a,n) )              # Term 1: triplet
      + lam_repel    * relu( d(a,p) + m2 - d(a,m) )     # Term 2: repel morph from A
      + lam_repel    * relu( d(a,p) + m2 - d(b,m) )     # Term 3: repel morph from B
      + lam_directed * relu( d(m, sg(n)) - d(a,m) )     # Term 4: directed A-side
      + lam_directed_b * relu( d(m, sg(n)) - d(b,m) )   # Term 5: directed B-side (optional)

    sg(n) = stop-gradient on impostor (only zm receives gradient from Terms 4–5).
    Term 5 is inactive when lam_directed_b=0.0 (default) — backward-compatible.

    Args:
        margin:         m1, triplet hinge margin (default: 0.5)
        margin_repel:   m2, morph repulsion hinge margin (default: 0.3)
        lam_repel:      weight on the two repulsion terms (default: 1.0)
        lam_directed:   weight on the A-side directed term (default: 0.3)
        lam_directed_b: weight on the B-side directed term (default: 0.0 = off)

    Forward inputs (quintuplet, all L2-normalised adapter outputs):
        za: contributor A anchor   [B, D]
        zp: positive (other image of A)  [B, D]
        zn: random non-contributor       [B, D]
        zm: morph                        [B, D]
        zb: contributor B anchor         [B, D]

    Returns (reduction="mean"):
        loss:             scalar total loss
        loss_triplet:     scalar Term 1
        loss_repel:       scalar Terms 2+3 combined (raw, before lam_repel)
        loss_directed_a:  scalar Term 4 (raw, before lam_directed)
        loss_directed_b:  scalar Term 5 (raw, before lam_directed_b)
    """

    def __init__(
        self,
        margin: float = 0.5,
        margin_repel: float = 0.3,
        lam_repel: float = 1.0,
        lam_directed: float = 0.3,
        lam_directed_b: float = 0.0,
    ):
        super().__init__()
        self.margin = margin
        self.margin_repel = margin_repel
        self.lam_repel = lam_repel
        self.lam_directed = lam_directed
        self.lam_directed_b = lam_directed_b

    def forward(self, za, zp, zn, zm, zb, reduction="mean"):
        """
        :param za: contributor A anchor embeddings (L2-normalized)
        :param zp: positive embeddings (different image of A, L2-normalized)
        :param zn: random non-contributor embeddings (L2-normalized)
        :param zm: morph embeddings (L2-normalized)
        :param zb: contributor B anchor embeddings (L2-normalized)
        :param reduction: "mean" or "none"
        """
        dap = euclid(za, zp)
        dan = euclid(za, zn)
        dam = euclid(za, zm)
        dbm = euclid(zb, zm)
        dmn = euclid(zm, zn.detach())   # stop-gradient on impostor

        loss_triplet    = F.relu(dap + self.margin - dan)
        loss_repel_a    = F.relu(dap + self.margin_repel - dam)
        loss_repel_b    = F.relu(dap + self.margin_repel - dbm)
        loss_repel      = loss_repel_a + loss_repel_b
        loss_directed_a = F.relu(dmn - dam)  # hinge: fires when d(m,n) > d(a,m)
        loss_directed_b = F.relu(dmn - dbm)  # hinge: fires when d(m,n) > d(b,m)

        loss = (
            loss_triplet
            + self.lam_repel     * loss_repel
            + self.lam_directed  * loss_directed_a
            + self.lam_directed_b * loss_directed_b
        )

        if reduction == "none":
            return loss, loss_triplet, loss_repel, loss_directed_a, loss_directed_b
        if reduction == "mean":
            return (loss.mean(), loss_triplet.mean(), loss_repel.mean(),
                    loss_directed_a.mean(), loss_directed_b.mean())
        raise ValueError(f"Unsupported reduction: {reduction}")


class TripletLoss(nn.Module):
    """
    Standard triplet margin loss in distance space:

        L_i = max(0, d(anchor, positive) + margin - d(anchor, negative))
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, xa, xp, xn, reduction="mean"):
        """
        :param xa: anchor embeddings (B, D)
        :param xp: positive embeddings (B, D)
        :param xn: negative embeddings (B, D)
        :param reduction: "mean" or "none"
        """
        dap = euclid(xa, xp)
        dan = euclid(xa, xn)

        loss = F.relu(dap + self.margin - dan)

        if reduction == "none":
            return loss
        if reduction == "mean":
            
            
            return loss.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")