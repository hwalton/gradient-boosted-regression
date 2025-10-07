#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./destroy_resources.sh        # default: uninstall release 'gbr' in namespace 'default', delete PVCs/PVs
#   KEEP_PVCS=1 ./destroy_resources.sh   # keep PVCs and PVs
#   DELETE_CRDS=1 ./destroy_resources.sh # also attempt to delete CRDs labeled for the release (use with care)

: "${RELEASE:=gbr}"
: "${NAMESPACE:=default}"
: "${KEEP_PVCS:=0}"     # set to 1 to preserve PVCs/PVs
: "${DELETE_CRDS:=0}"   # set to 1 to delete CRDs that match label (dangerous)

echo "Destroying resources for release=$RELEASE namespace=$NAMESPACE (keep_pvcs=$KEEP_PVCS delete_crds=$DELETE_CRDS)"

# 1) Try helm uninstall first (safe, preferred)
if command -v helm >/dev/null 2>&1; then
  echo "Uninstalling Helm release ${RELEASE} in ${NAMESPACE} (if present)..."
  helm uninstall "${RELEASE}" -n "${NAMESPACE}" || true

  # wait a short time for helm to remove resources
  sleep 5
fi

# 2) Delete namespaced resources created by the release (by common Helm labels)
LABEL="app.kubernetes.io/instance=${RELEASE}"
echo "Deleting namespaced resources with label ${LABEL} in ${NAMESPACE}..."
kubectl delete all,configmap,secret,service,ingress,job,cronjob,deploy,statefulset,daemonset,replicaset --ignore-not-found -l "${LABEL}" -n "${NAMESPACE}" || true

# 3) Delete other namespaced objects (secrets/configmaps) with that label across all namespaces
echo "Deleting any other namespaced resources labelled ${LABEL} across all namespaces..."
kubectl get role,rolebinding,pvc -A -l "${LABEL}" -o name --ignore-not-found \
  | xargs -r -n1 kubectl delete --ignore-not-found

# 4) Optionally delete PVCs and their PVs in the target namespace
if [ "${KEEP_PVCS}" -eq 0 ]; then
  echo "Deleting PVCs in namespace ${NAMESPACE} with label ${LABEL}..."
  PVCS=$(kubectl get pvc -n "${NAMESPACE}" -l "${LABEL}" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' || true)
  if [ -n "${PVCS}" ]; then
    echo "${PVCS}" | xargs -r -n1 -I{} sh -c 'kubectl delete pvc -n "'"${NAMESPACE}"'" "{}" --ignore-not-found || true'
    # delete PVs whose claimRef points to namespace/name (only PVs bound to that namespace)
    echo "Deleting PVs bound to PVCs from namespace ${NAMESPACE}..."
    kubectl get pv -o jsonpath='{range .items[?(@.spec.claimRef.namespace=="'"${NAMESPACE}"'")]}{.metadata.name}{"\n"}{end}' \
      | xargs -r -n1 kubectl delete pv --ignore-not-found || true
  else
    echo "No PVCs found for ${LABEL} in ${NAMESPACE}."
  fi
else
  echo "KEEP_PVCS set: preserving PVCs/PVs."
fi

# 5) Optionally remove CRDs labelled for the release (cluster-scoped) — use with care
if [ "${DELETE_CRDS}" -eq 1 ]; then
  echo "Deleting CRDs that match label ${LABEL} (cluster-scoped) -- PROCEED WITH CAUTION"
  kubectl get crd -l "${LABEL}" -o name --ignore-not-found \
    | xargs -r -n1 kubectl delete --ignore-not-found
fi

# 6) Remove cluster-scoped objects created by Helm (ClusterRoles, ClusterRoleBindings, CRs labelled)
echo "Deleting cluster-scoped ClusterRole/ClusterRoleBinding/CustomResources labelled ${LABEL}..."
kubectl get clusterrole,clusterrolebinding -l "${LABEL}" -o name --ignore-not-found \
  | xargs -r -n1 kubectl delete --ignore-not-found

# Delete any remaining resources (defensive)
echo "Cleaning any remaining objects labelled ${LABEL} across all namespaces..."
kubectl get all -A -l "${LABEL}" -o name --ignore-not-found \
  | xargs -r -n1 kubectl delete --ignore-not-found

# 7) Wait for pods to terminate
echo "Waiting for pods with label ${LABEL} to be removed..."
kubectl wait --for=delete pod -l "${LABEL}" -n "${NAMESPACE}" --timeout=120s || true

echo "Done. Resources for release=${RELEASE} removed (cluster preserved)."