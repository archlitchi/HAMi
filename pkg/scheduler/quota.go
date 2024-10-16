/*
Copyright 2024 The HAMi Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package scheduler

import (
	"strings"
	"sync"

	"github.com/Project-HAMi/HAMi/pkg/device/nvidia"
	"github.com/Project-HAMi/HAMi/pkg/util"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

type Quota struct {
	Used  int64
	Limit int64
}

type DeviceQuota map[string]*Quota

type quotaManager struct {
	Quotas map[string]*DeviceQuota
	mutex  sync.RWMutex
}

func (q *quotaManager) init() {
	q.Quotas = make(map[string]*DeviceQuota)
}

func (q *quotaManager) FitQuota(ns string, memreq int64, coresreq int64) bool {
	dq := q.Quotas[ns]
	if dq == nil {
		return true
	}
	klog.InfoS("resourceMem quota judging", "limit", (*dq)[nvidia.ResourceMem].Limit, "used", (*dq)[nvidia.ResourceMem].Used, "alloc", memreq)
	if (*dq)[nvidia.ResourceMem].Limit != 0 && (*dq)[nvidia.ResourceMem].Used+memreq > (*dq)[nvidia.ResourceMem].Limit {
		klog.InfoS("resourceMem quota not fitted", "limit", (*dq)[nvidia.ResourceMem].Limit, "used", (*dq)[nvidia.ResourceMem].Used, "alloc", memreq)
		return false
	}
	if (*dq)[nvidia.ResourceCores].Limit != 0 && (*dq)[nvidia.ResourceCores].Used+coresreq > (*dq)[nvidia.ResourceCores].Limit {
		klog.InfoS("resourceCores quota not fitted", "limit", (*dq)[nvidia.ResourceCores].Limit, "used", (*dq)[nvidia.ResourceCores].Used, "alloc", memreq)
		return false
	}
	return true
}

func countPodDevices(podDev util.PodDevices) map[string]int64 {
	res := make(map[string]int64)
	for deviceName, podSingle := range podDev {
		if !strings.Contains(deviceName, "NVIDIA") {
			continue
		}
		for _, ctrdevices := range podSingle {
			for _, ctrdevice := range ctrdevices {
				res[nvidia.ResourceMem] += int64(ctrdevice.Usedmem)
				res[nvidia.ResourceCores] += int64(ctrdevice.Usedcores)
			}
		}
	}
	return res
}

func (q *quotaManager) addUsage(pod *corev1.Pod, podDev util.PodDevices) {
	usage := countPodDevices(podDev)
	if len(usage) == 0 {
		return
	}
	if q.Quotas[pod.Namespace] == nil {
		q.Quotas[pod.Namespace] = &DeviceQuota{}
	}
	dp := q.Quotas[pod.Namespace]
	for idx, val := range usage {
		_, ok := (*dp)[idx]
		if !ok {
			(*dp)[idx] = &Quota{
				Used:  0,
				Limit: 0,
			}
		}
		(*dp)[idx].Used += val
	}
	for _, val := range q.Quotas {
		for idx, val1 := range *val {
			klog.Infoln("after val=", idx, ":", val1)
		}
	}
}

func (q *quotaManager) rmUsage(pod *corev1.Pod, podDev util.PodDevices) {
	usage := countPodDevices(podDev)
	if len(usage) == 0 {
		return
	}
	dp := q.Quotas[pod.Namespace]
	for idx, val := range usage {
		(*dp)[idx].Used -= val
	}
	for _, val := range q.Quotas {
		for idx, val1 := range *val {
			klog.Infoln("after val=", idx, ":", val1)
		}
	}
}

func (q *quotaManager) addQuota(quota *corev1.ResourceQuota) {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	for idx, val := range quota.Spec.Hard {
		value, ok := val.AsInt64()
		if ok {
			dn := idx.String()[len("requests."):]
			if !strings.Contains(dn, nvidia.ResourceMem) && !strings.Contains(dn, nvidia.ResourceCores) {
				continue
			}
			if q.Quotas[quota.Namespace] == nil {
				q.Quotas[quota.Namespace] = &DeviceQuota{}
			}
			dp := q.Quotas[quota.Namespace]
			_, ok := (*dp)[dn]
			if !ok {
				(*dp)[dn] = &Quota{
					Used:  0,
					Limit: value,
				}
			}
			(*dp)[dn].Limit = value
			klog.InfoS("quota set:", "idx=", idx, "val", value)
		}
	}
	for _, val := range q.Quotas {
		for idx, val1 := range *val {
			klog.Infoln("after val=", idx, ":", val1)
		}
	}
}

func (q *quotaManager) delQuota(quota *corev1.ResourceQuota) {
	q.mutex.Lock()
	defer q.mutex.Unlock()
	for idx, val := range quota.Spec.Hard {
		value, ok := val.AsInt64()
		if ok {
			dn := idx.String()[len("requests."):]
			if !strings.Contains(dn, nvidia.ResourceMem) && !strings.Contains(dn, nvidia.ResourceCores) {
				continue
			}
			klog.InfoS("quota remove:", "idx=", idx, "val", value)
			(*q.Quotas[quota.Namespace])[dn].Limit = 0
		}
	}
	for _, val := range q.Quotas {
		for idx, val1 := range *val {
			klog.Infoln("after val=", idx, ":", val1)
		}
	}

}
